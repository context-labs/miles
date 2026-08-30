"""Convert fable teacher trajectories (Anthropic-ingest JSONL) to Miles SFT parquet.

Each input line is one session record; ``training_messages_json.messages`` holds an
OpenAI-normalized conversation (system/user/assistant/tool, assistant ``tool_calls``
with JSON-string arguments). Sessions are far too long for full-sequence SFT
(median ~430K tokens), so this script emits **per-assistant-turn** samples: for each
selected assistant turn, the context is the first user message (the task) plus a
greedy tail window of turns that fits ``--max-tokens``; the teacher's system
message is dropped (it is a ~250K-token deployment-specific brief — noise for
transfer). The loss mask covers only the final assistant turn. This mirrors how the model is used at
serving time (task + recent context -> next action).

Rendering uses DeepSeek-V4's canonical encoder (``sglang...encoding_dsv4``): tool
messages merged into user messages, tool results sorted by call order, historical
thinking dropped — exactly what SGLang serves. Every sample is verified: our
per-message concatenation must equal ``encode_messages`` of the same list, and the
per-message token spans must tile the full tokenization exactly; failures are
skipped and counted, never silently corrupted.

Output parquet columns:
  - ``messages``: JSON string of the (truncated, normalized) conversation — the
    ``--input-key`` column, used by Miles for logging/debugging only.
  - ``metadata``: struct with ``tokens_json`` / ``loss_mask_json`` (JSON strings of
    int lists) consumed by ``sft_rollout.py`` in this directory.

Usage (inside the miles container):

    python examples/deepseek_v4_sft_fable/prep_data.py \
        --input /data/fable_latest_per_session.jsonl \
        --hf-checkpoint /models-v4/DeepSeek-V4-Flash-FP8 \
        --output /data/fable_sft_v4.parquet

    # quick stats over a prefix of the file, writing nothing:
    python examples/deepseek_v4_sft_fable/prep_data.py --input ... --hf-checkpoint ... \
        --analyze-only --limit 200
"""

import argparse
import json
import logging
import multiprocessing as mp
from typing import Any

logger = logging.getLogger(__name__)

# Set per worker process in _init_worker.
_TOKENIZER = None


def _flatten_content(content: Any) -> str:
    """Join list-form content blocks into one string; pass strings through."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or "")
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _has_images(messages: list[dict]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "image_url" for b in c):
            return True
    return False


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Flatten content and keep only fields the V4 encoder understands."""
    out = []
    for m in messages:
        msg = {"role": m["role"], "content": _flatten_content(m.get("content"))}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if m.get("reasoning_content"):
            msg["reasoning_content"] = _flatten_content(m["reasoning_content"])
        out.append(msg)
    return out


def _render_messages(messages: list[dict]) -> tuple[list[str], list[str], str] | None:
    """Render each message with the V4 encoder; verify against the canonical render.

    Returns (per-message texts, roles, full text) or None on any mismatch.
    """
    from sglang.srt.entrypoints.openai import encoding_dsv4

    merged = encoding_dsv4.merge_tool_messages(messages)
    merged = encoding_dsv4.sort_tool_results_by_call_order(merged)
    drop_thinking = not any(m.get("tools") for m in merged)
    if drop_thinking:
        merged = encoding_dsv4._drop_thinking_messages(merged)

    per_message = []
    roles = []
    for idx, msg in enumerate(merged):
        per_message.append(
            encoding_dsv4.render_message(idx, merged, thinking_mode="thinking", drop_thinking=drop_thinking)
        )
        roles.append(msg.get("role"))
    full_text = encoding_dsv4.bos_token + "".join(per_message)

    canonical = encoding_dsv4.encode_messages(messages, thinking_mode="thinking")
    if full_text != canonical:
        return None
    return per_message, roles, full_text


def _tokenize_with_spans(per_message: list[str], full_text: str):
    """Tokenize once and map message char spans to token spans via offset mapping.

    Message boundaries in the V4 format fall on special tokens, so a boundary
    landing mid-token means BPE merged across it — the sample is rejected.
    """
    encoding = _TOKENIZER(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # full_text = bos_token + messages; per-message spans start after the BOS chars.
    from sglang.srt.entrypoints.openai import encoding_dsv4

    spans = []
    char_cursor = len(encoding_dsv4.bos_token)
    tok_cursor = 0
    for text in per_message:
        char_start, char_end = char_cursor, char_cursor + len(text)
        while tok_cursor < len(offsets) and offsets[tok_cursor][1] <= char_start:
            tok_cursor += 1
        tok_start = tok_cursor
        if tok_start < len(offsets) and offsets[tok_start][0] < char_start:
            return None, None  # token straddles the message boundary
        while tok_cursor < len(offsets) and offsets[tok_cursor][1] <= char_end:
            tok_cursor += 1
        spans.append((tok_start, tok_cursor))
        char_cursor = char_end
    if char_cursor != len(full_text):
        return None, None
    return full_ids, spans


def _turn_samples(messages: list[dict], max_tokens: int, max_turns_per_session: int, min_tokens: int) -> list[dict]:
    """Emit one sample per selected assistant turn, with a token-bounded context window."""
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_idx:
        return []
    # Prefer recent turns: they carry the most task progress.
    selected = assistant_idx[-max_turns_per_session:]

    # The teacher's system message is a ~250K-token deployment-specific brief
    # (internal runbooks/credentials) — pure noise for transfer and blows any
    # context budget. Train on task + turns only.
    messages = [m for m in messages if m["role"] != "system"]

    samples = []
    for tail_idx in reversed(selected):
        first_user = next((m for m in messages if m["role"] == "user"), None)
        head = [first_user] if first_user is not None else []
        head_idx = {id(m) for m in head}

        window = [m for i, m in enumerate(messages[: tail_idx + 1]) if id(m) in head_idx or i > 0]
        # Greedy left-truncation: drop oldest non-head turns until the render fits.
        while True:
            rendered = _render_messages(window)
            if rendered is None:
                break
            per_message, roles, full_text = rendered
            approx_tokens = len(full_text) // 3
            if approx_tokens <= max_tokens * 1.15:  # close enough to afford exact tokenization
                full_ids, spans = _tokenize_with_spans(per_message, full_text)
                if full_ids is None:
                    break
                if len(full_ids) <= max_tokens and len(full_ids) >= min_tokens:
                    loss_mask = [0] * len(full_ids)
                    start, end = spans[-1]
                    loss_mask[start:end] = [1] * (end - start)
                    samples.append({"messages": window, "tokens": full_ids, "loss_mask": loss_mask})
                elif len(full_ids) > max_tokens and len(window) <= len(head) + 1:
                    break  # head + final turn alone exceeds the budget; give up on this turn
            # shrink the window: drop the oldest message that is neither head nor
            # the final turn; as a last resort drop the head too (tail-only window)
            removable = [i for i, m in enumerate(window[:-1]) if id(m) not in head_idx]
            if not removable:
                removable = list(range(len(window) - 1))
            if not removable:
                break
            window.pop(removable[0])
    return samples


def _process_session(record: dict, max_tokens: int, max_turns_per_session: int, min_tokens: int) -> list[dict]:
    messages = record["training_messages_json"]["messages"]
    if len(messages) < 2 or not any(m["role"] == "assistant" for m in messages) or _has_images(messages):
        return []
    messages = _normalize_messages(messages)
    try:
        return _turn_samples(messages, max_tokens, max_turns_per_session, min_tokens)
    except Exception as e:
        logger.debug("session %s failed: %s", record.get("id"), e)
        return []


def _init_worker(hf_checkpoint: str):
    global _TOKENIZER
    from miles.utils.processing_utils import load_tokenizer

    _TOKENIZER = load_tokenizer(hf_checkpoint, trust_remote_code=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="fable JSONL (one session per line)")
    parser.add_argument("--hf-checkpoint", required=True, help="V4 checkpoint (for the tokenizer)")
    parser.add_argument("--output", default="", help="output parquet path")
    parser.add_argument("--max-tokens", type=int, default=16384, help="per-sample token budget")
    parser.add_argument("--min-tokens", type=int, default=256, help="drop degenerate tiny samples")
    parser.add_argument("--max-turns-per-session", type=int, default=12)
    parser.add_argument("--analyze-only", action="store_true", help="print stats, write nothing")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N lines")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    lines = []
    with open(args.input) as f:
        for line in f:
            lines.append(line)
            if args.limit is not None and len(lines) >= args.limit:
                break
    logger.info("loaded %d lines", len(lines))

    samples = []
    sessions_used = 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.hf_checkpoint,)) as pool:
        for result in pool.starmap(
            _process_line,
            [(line, args.max_tokens, args.max_turns_per_session, args.min_tokens) for line in lines],
            chunksize=1,
        ):
            if result:
                samples.extend(result)
                sessions_used += 1
    logger.info("sessions with >=1 sample: %d / %d; total samples: %d", sessions_used, len(lines), len(samples))

    if samples:
        lens = sorted(len(s["tokens"]) for s in samples)
        n = len(lens)
        logger.info(
            "sample tokens: p50=%d p90=%d max=%d | total train tokens=%d",
            lens[n // 2],
            lens[int(n * 0.9)],
            lens[-1],
            sum(lens),
        )

    if args.analyze_only or not args.output:
        return

    import pandas as pd

    df = pd.DataFrame(
        {
            "messages": [json.dumps(s["messages"]) for s in samples],
            "metadata": [
                {"tokens_json": json.dumps(s["tokens"]), "loss_mask_json": json.dumps(s["loss_mask"])}
                for s in samples
            ],
        }
    )
    df.to_parquet(args.output)
    logger.info("wrote %s (%d rows)", args.output, len(df))


def _process_line(line: str, max_tokens: int, max_turns_per_session: int, min_tokens: int) -> list[dict]:
    return _process_session(json.loads(line), max_tokens, max_turns_per_session, min_tokens)


if __name__ == "__main__":
    main()
