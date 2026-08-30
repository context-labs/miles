"""SFT rollout for pre-tokenized DeepSeek-V4 fable data.

Companion to ``prep_data.py``: the parquet already carries exact ``tokens`` and
``loss_mask`` produced with the canonical V4 encoder, so this function only moves
them onto the samples — no chat-template machinery at train time.

Sample contract mirrors ``miles.rollout.sft_rollout.generate_rollout``:
``response_length`` runs from the first trainable token to the end of the
sequence, and ``loss_mask`` is trimmed to that region (zeros inside it are
allowed; they mark non-assistant turns inside the training window).
"""

import json
import logging

from miles.utils.mask_utils import get_response_lengths

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

_SAMPLE_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global _SAMPLE_PRINTED
    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        tokens = json.loads(sample.metadata["tokens_json"])
        loss_mask = json.loads(sample.metadata["loss_mask_json"])
        assert len(tokens) == len(loss_mask)

        sample.tokens = tokens
        sample.response_length = get_response_lengths([loss_mask])[0]
        sample.loss_mask = loss_mask[-sample.response_length :]
        sample.reward = 0

        if i == 0 and not _SAMPLE_PRINTED:
            logger.info(
                f"fable sft_rollout example: tokens={len(tokens)} response_length={sample.response_length} "
                f"train_tokens={sum(sample.loss_mask)}"
            )
            _SAMPLE_PRINTED = True

    return samples
