"""DeepSeek-V4-Flash LoRA SFT on fable teacher trajectories (single node, 8 GPUs).

Pure SFT: runs ``train_async.py`` with ``--debug-train-only`` — no SGLang engine,
no generation, no eval. The dataset is pre-tokenized by
``examples/deepseek_v4_sft_fable/prep_data.py`` (canonical V4 encoder rendering +
verified loss masks), and ``sft_rollout.py`` in the same directory just moves the
precomputed tensors onto samples.

Prerequisites:
  - BF16 cast + torch_dist conversion of sgl-project/DeepSeek-V4-Flash-FP8
    (see scripts/run_deepseek_v4.py prepare-single / prepare-spmd).
  - Miles checkout with the V4 LoRA stack (PR #2706 + #2772 rebase) and Megatron
    at radixark/Megatron-LM:dsv4-dual-backend.

Usage:
    python examples/deepseek_v4_sft_fable/run.py \
        --model-dir /var/tmp/v4-models \
        --prompt-data /data/fable_sft_v4.parquet
"""

import os
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    megatron_model_type: str = "deepseek-v4-flash"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    model_dir: str = "/root/models"
    model_name: str = "DeepSeek-V4-Flash-FP8"
    save_dir: str = ""
    prompt_data: str = "/root/datasets/fable_sft_v4.parquet"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "wq_a,wq_b,wkv,wo_b"

    # Training
    num_epoch: int = 2
    rollout_batch_size: int = 64
    global_batch_size: int = 64
    max_seq_len: int = 17408
    lr: float = 1e-5
    save_interval: int = 50
    extra_args: str = ""

    @property
    def model_path(self) -> str:
        return f"{self.model_dir}/{self.model_name}"


def execute(args: ScriptArgs):
    save_dir = args.save_dir or f"{args.model_dir}/runs/{args.run_id}"

    ckpt_args = (
        f"--hf-checkpoint {args.model_path} "
        f"--ref-load {args.model_path}_torch_dist "
        f"--load {save_dir}/checkpoints "
        f"--save {save_dir}/checkpoints "
        f"--save-interval {args.save_interval} "
    )

    lora_args = (
        "--megatron-to-hf-mode bridge "
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f"--target-modules {args.target_modules} "
        "--no-gradient-accumulation-fusion "
    )

    sft_args = (
        "--rollout-function-path sft_rollout.generate_rollout "
        f"--prompt-data {args.prompt_data} "
        "--input-key messages "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-epoch {args.num_epoch} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--global-batch-size {args.global_batch_size} "
        f"--seq-length {args.max_seq_len} "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        # pure SFT: no sglang engine, no generation
        "--debug-train-only "
    )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # bshd (miles) path: micro-batch 1, dynamic batching is thd-only
        "--micro-batch-size 1 "
        f"--max-tokens-per-gpu {args.max_seq_len} "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style cosine "
        "--min-lr 1e-6 "
        "--lr-warmup-fraction 0.1 "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    v4_args = (
        "--dsv4-impl miles "
        "--qkv-format bshd "
        "--model-name deepseekv4 "  # for mbridge load
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} {lora_args} {sft_args} {optimizer_args} {perf_args} {v4_args} {misc_args} {args.extra_args} "
    )

    miles_root = U.repo_base_dir
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={
            "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
