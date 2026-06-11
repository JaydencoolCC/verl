#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | FSDP training | NVIDIA GPUs

set -xeuo pipefail

export WANDB_PROJECT="verl_opd_gsm8k"
export WANDB_ENTITY="jaycool"
export WANDB_API_KEY="52c080a583ec102ea25df24a115fc0e0678aca35"
export HF_ENDPOINT=https://hf-mirror.com

# This script targets NVIDIA/CUDA. Clear ROCm device masks that can be inherited
# from the shell and conflict with Ray's per-worker CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# ---- user-adjustable ----

# Models.
# Student model to train. Can be a Hugging Face model id or a local path.
# STUDENT_MODEL="/data/common/LLMs/Qwen3-1.7B"
STUDENT_MODEL="/data/common/LLMs/Qwen3-1.7B-Base"
# Teacher model used only for inference/log-prob scoring during distillation.
TEACHER_MODEL="/data/common/LLMs/Qwen3-4B"

# Resources.
# Number of training nodes.
NNODES=1
# Number of GPUs per node used by actor training and student rollout.
NGPUS_PER_NODE=4
# Tensor-parallel size for the student rollout vLLM engine.
rollout_tp=1
# Number of GPUs per node reserved for the teacher inference resource pool.
TEACHER_WORLD_SIZE=1
# Tensor-parallel size for each teacher vLLM replica.
teacher_tp=1
# Number of async agent-loop workers used to dispatch rollout requests.
rollout_agent_workers=${NGPUS_PER_NODE}

# Sampling.
# Temperature for student rollout sampling.
rollout_temperature=1.0
# Temperature for teacher log-prob scoring.
teacher_temperature=1.0
# Temperature for validation/evaluation sampling.
val_temperature=0.0
# Whether validation/evaluation uses sampling instead of greedy decoding.
val_do_sample=False
# Repetition penalty for student rollout sampling; 1.0 means no penalty.
rollout_repetition_penalty=1.0


# Inference engine memory.
# Fraction of each rollout GPU's memory vLLM may use for KV cache.
rollout_gpu_mem_util=0.4
# Fraction of each teacher GPU's memory vLLM may use for KV cache.
teacher_gpu_mem_util=0.8

# Distillation.
# Loss type. Available values:
#   k1: sampled-token KL estimator, same formula as kl.
#   kl: student_logprob - teacher_logprob for sampled tokens.
#   abs: absolute difference between student and teacher logprob.
#   mse: squared logprob difference, same formula as k2.
#   k2: 0.5 * squared logprob difference.
#   low_var_kl: low-variance KL estimator, same formula as k3.
#   k3: low-variance KL estimator for sampled tokens.
#   forward_kl_topk: forward KL over teacher top-k token probabilities.

# - **Top-k** (`forward_kl_topk`): forward KL using the teacher's top-k logits.
# - **Single-sample KL estimators** (`kl`, `k1`, `abs`, `mse`, `k2`,
#   `low_var_kl`, `k3`): per-token Monte Carlo estimators of reverse KL
#   computed from the student's `log_probs` and the teacher's single
#   `log_prob` at the sampled token.

distillation_loss_mode="k1"
# Turn off policy-gradient mode when using top-k distillation losses.
use_policy_gradient=True
# Number of teacher top tokens used by top-k distillation modes.
distillation_topk=4

# Batch and sequence lengths.
# Global train batch size sampled per PPO/GRPO training iteration.
train_batch_size=32
# PPO mini-batch size used when updating the actor.
ppo_mini_batch_size=32
# Maximum prompt length in tokens.
max_prompt_length=1024
# Maximum generated response length in tokens.
# max_response_length=8172
max_response_length=4096
# Token budget per GPU for dynamic actor/log-prob batches; lower it if training OOMs.
# ppo_max_token_len_per_gpu=32768
ppo_max_token_len_per_gpu=16384

# Optimization.
# Actor optimizer learning rate.
actor_lr=1e-6

# Training schedule.
# Number of full passes over the training dataset.
total_epochs=2
# Save checkpoint every N training steps.
save_freq=82
# Run validation every N training steps.
test_freq=20

# Logging and outputs.
# Experiment tracking project name.
project_name="verl_distill_contamination"
# Experiment tracking run name.
experiment_name="qwen3_1.7B_2_4B_k1"
# experiment_name="qwen3_1.test"
# Directory for saving decoded student rollout generations as JSONL.
rollout_data_dir="/data/home/zhanghx/toy_try/verl/rollout_generations/${experiment_name}"
# Directory for saving decoded validation generations as JSONL.
validation_data_dir="/data/home/zhanghx/toy_try/verl/validation_generations/${experiment_name}"
# Number of validation generations to log to experiment trackers.
log_val_generations=8

# Data.
# GSM8K training split.
gsm8k_train=/data/home/zhanghx/dataset/verl/gsm8k/train.parquet
# GSM8K validation split.
gsm8k_test=/data/home/zhanghx/dataset/verl/gsm8k/test.parquet
# MATH500 validation split.
math500_test=/data/home/zhanghx/dataset/verl/math500/test.parquet
# DAPO training split.
dapo_train=/data/home/zhanghx/dataset/verl/dapo-math-17k/train.parquet

# # MATH training split.
# math_train=$HOME/data/math/train.parquet
# # MATH validation split.
# math_test=$HOME/data/math/test.parquet

# Hydra list literal containing all training parquet files.
# train_files="['$dapo_train']"
train_files="['$gsm8k_test']"
# Hydra list literal containing all validation parquet files.
val_files="['$gsm8k_test']"

# ---- end user-adjustable ----

# Required context length for rollout and teacher scoring: prompt + response + one extra token.
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    # Use GRPO advantage estimation.
    algorithm.adv_estimator=grpo
    # Disable KL penalty inside reward; distillation provides the main learning signal here.
    algorithm.use_kl_in_reward=False
    # Training parquet files.
    data.train_files="$train_files"
    # Validation parquet files.
    data.val_files="$val_files"
    # Global batch size per training iteration.
    data.train_batch_size=${train_batch_size}
    # Maximum allowed prompt length.
    data.max_prompt_length=${max_prompt_length}
    # Maximum generated response length.
    data.max_response_length=${max_response_length}
    # Drop prompts that exceed max_prompt_length during dataset loading.
    data.filter_overlong_prompts=True
    # Raise an error instead of truncating if an overlong prompt reaches tokenization.
    data.truncation='error'
    # Disable Qwen3 thinking mode when applying the chat template.
    +data.apply_chat_template_kwargs.enable_thinking=False
    # Keep dataset order deterministic.
    data.shuffle=False
    data.dataloader_num_workers=4
)

MODEL=(
    # Student model path/id used by actor, rollout, and reference components.
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    # Remove padding tokens before model forward to reduce compute and memory.
    actor_rollout_ref.model.use_remove_padding=True
    # Trade extra recomputation for lower activation memory during training.
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    # Enable torch.compile for actor training.
    actor_rollout_ref.actor.use_torch_compile=True
    # actor_rollout_ref.actor.use_torch_compile=False
    # Actor learning rate.
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    # PPO mini-batch size for actor updates.
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    # Pack work by token budget instead of fixed micro-batch size.
    actor_rollout_ref.actor.use_dynamic_bsz=True
    # Maximum valid tokens processed per GPU in one dynamic actor batch.
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Keep FSDP parameters and optimizer states on GPU for faster training.
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    # Use vLLM for student response generation.
    actor_rollout_ref.rollout.name=vllm
    # Skip vLLM CUDA graph capture for faster startup during rollout initialization.
    actor_rollout_ref.rollout.enforce_eager=False
    # Must divide data.train_batch_size because DataProto chunks rollout prompts equally.
    actor_rollout_ref.rollout.agent.num_workers=${rollout_agent_workers}
    # Tensor-parallel size for the rollout engine.
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    # Temperature for student response sampling.
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    # Sampling settings for validation/evaluation.
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}
    +actor_rollout_ref.rollout.repetition_penalty=${rollout_repetition_penalty}
    # Fraction of GPU memory reserved for rollout KV cache.
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    # Number of sampled responses per prompt.
    actor_rollout_ref.rollout.n=1
    # Maximum context length accepted by the rollout engine.
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    # Maximum total tokens vLLM may batch in one rollout scheduling step.
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    # Use dynamic token batching when recomputing rollout log-probs.
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    # Token budget per GPU for rollout log-prob computation.
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}

)

TRAINER=(
    # Reorder batch items across workers to balance sequence lengths and reduce stragglers.
    trainer.balance_batch=True
    # Log both to stdout and Weights & Biases.
    # trainer.logger='["console"]'
    trainer.logger='["console","wandb"]'
    # Tracking project name.
    trainer.project_name=${project_name}
    # Tracking experiment/run name.
    trainer.experiment_name=${experiment_name}
    # Save decoded student rollout generations for inspection.
    trainer.rollout_data_dir=${rollout_data_dir}
    # Save decoded validation generations for inspection.
    trainer.validation_data_dir=${validation_data_dir}
    # Log a small validation generation table to W&B.
    # trainer.log_val_generations=${log_val_generations}
    # Number of training GPUs on each node.
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    # Number of training nodes.
    trainer.nnodes=${NNODES}
    # Run validation before the first training step to capture the initial student behavior.
    trainer.val_before_train=True
    # Always start from the base student model instead of auto-resuming a checkpoint.
    trainer.resume_mode=disable
    # Checkpoint save interval in training steps.
    trainer.save_freq=${save_freq}
    # Validation interval in training steps.
    trainer.test_freq=${test_freq}
    # Total number of training epochs.
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    # Enable on-policy distillation and allocate teacher workers.
    distillation.enabled=True
    # GPUs per node in the teacher resource pool.
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    # Number of nodes in the teacher resource pool.
    distillation.nnodes=${NNODES}
    # Teacher model path/id used for inference scoring.
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    # Tensor-parallel size for each teacher inference replica.
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    # Temperature for teacher log-prob scoring.
    distillation.teacher_models.teacher_model.inference.temperature=${teacher_temperature}
    # Use vLLM as the teacher inference engine.
    distillation.teacher_models.teacher_model.inference.name=vllm
    # Fraction of teacher GPU memory reserved for vLLM KV cache.
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    # Teacher context length; must fit student prompt + full response + one extra token.
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    # Distillation divergence/estimator to optimize.
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    # Top-k size for loss modes that use teacher top-k log-probs.
    distillation.distillation_loss.topk=${distillation_topk}
    # Do not add normal task reward PPO/GRPO loss; train only on distillation signal.
    distillation.distillation_loss.use_task_rewards=False
    # If true, use distillation loss as a policy-gradient reward; otherwise backprop it directly.
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    # Clamp per-token distillation loss to [-10, 10] for numerical stability.
    distillation.distillation_loss.loss_max_clamp=10.0
    # Clamp log-probabilities from below to avoid extreme KL values.
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
