# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    sampled_token_ids: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk + 1). The final column stores the sampled-token log prob.
        teacher_topk_ids: (bsz, seqlen, topk + 1). The final column stores the sampled-token id or a dummy id.
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)
    topk = config.distillation_loss.topk
    assert topk is not None
    teacher_topk_log_probs = teacher_topk_log_probs[..., :topk]
    teacher_topk_ids = teacher_topk_ids[..., :topk]

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }

def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    sampled_token_ids: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reverse KL distillation loss using teacher top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk + 1). The final column stores the sampled-token log prob.
        teacher_topk_ids: (bsz, seqlen, topk + 1). The final column stores the sampled-token id or a dummy id.
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)
    topk = config.distillation_loss.topk
    assert topk is not None
    sampled_teacher_log_probs = teacher_topk_log_probs[..., topk : topk + 1]
    teacher_topk_log_probs = teacher_topk_log_probs[..., :topk]
    teacher_topk_ids = teacher_topk_ids[..., :topk]

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
        sampled_teacher_log_probs = slice_input_tensor(sampled_teacher_log_probs, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]
    assert sampled_teacher_log_probs.shape[:2] == student_logits.shape[:2]
    assert sampled_teacher_log_probs.shape[-1] == 1
    assert sampled_token_ids is not None
    assert sampled_token_ids.shape == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    debug_student_topk_probs = student_topk_log_probs.exp()
    debug_teacher_topk_probs = teacher_topk_log_probs.exp()
    overlap_token_kl = student_topk_log_probs.exp() * (student_topk_log_probs - teacher_topk_log_probs)
    accumulated_count = torch.zeros_like(student_mass, dtype=torch.long)
    # TODO: chose loss
    loss_type = "kltopk_approx"
    if loss_type == "ukl":
        student_topk_probs = student_topk_log_probs.exp()
        teacher_topk_probs = teacher_topk_log_probs.exp()
        token_kl = student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)
        token_kl = token_kl - student_topk_probs + teacher_topk_probs
        distillation_losses = token_kl.sum(dim=-1)
    elif loss_type == "kltopk_mass":
        student_topk_log_probs = student_topk_log_probs - torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
        teacher_topk_log_probs = teacher_topk_log_probs - torch.logsumexp(teacher_topk_log_probs, dim=-1, keepdim=True)
        student_topk_probs = student_topk_log_probs.exp()
        token_kl = student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)
        topk_mass_loss_coef = 1.0
        mass_loss = (student_mass - teacher_mass).pow(2)
        distillation_losses = token_kl.sum(dim=-1) + topk_mass_loss_coef * mass_loss
    elif loss_type == "accumulated_kltopk_mass":
        threshold = 0.95
        teacher_topk_probs = teacher_topk_log_probs.exp()
        student_topk_probs = student_topk_log_probs.exp()
        accumulated_mask = (teacher_topk_probs.cumsum(dim=-1) - teacher_topk_probs) < threshold
        accumulated_student_mass = (student_topk_probs * accumulated_mask).sum(dim=-1, keepdim=True)
        accumulated_teacher_mass = (teacher_topk_probs * accumulated_mask).sum(dim=-1, keepdim=True)
        student_topk_log_probs = student_topk_log_probs - accumulated_student_mass.log()
        teacher_topk_log_probs = teacher_topk_log_probs - accumulated_teacher_mass.log()
        student_topk_probs = student_topk_log_probs.exp() * accumulated_mask
        token_kl = student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)
        topk_mass_loss_coef = 1.0
        mass_loss = (accumulated_student_mass - accumulated_teacher_mass).squeeze(-1).pow(2)
        distillation_losses = token_kl.sum(dim=-1) + topk_mass_loss_coef * mass_loss
        accumulated_count = accumulated_mask.sum(dim=-1)
    elif loss_type == "kltopk":
        student_topk_log_probs = student_topk_log_probs - torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
        teacher_topk_log_probs = teacher_topk_log_probs - torch.logsumexp(teacher_topk_log_probs, dim=-1, keepdim=True)
        student_topk_probs = student_topk_log_probs.exp()
        token_kl = student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)
        distillation_losses = token_kl.sum(dim=-1) 
    elif loss_type == "kltopk_unnormalized":
        student_topk_probs = student_topk_log_probs.exp()
        token_kl = student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)
        distillation_losses = token_kl.sum(dim=-1)
    elif loss_type == "kltopk_approx":
        sampled_token_ids_col = sampled_token_ids.unsqueeze(-1)
        sampled_not_in_topk = ~(teacher_topk_ids == sampled_token_ids_col).any(dim=-1, keepdim=True)

        # Build A' = teacher top-k tokens union sampled token.
        # The sampled-token slot is kept for a static tensor shape, and masked out
        # when the sampled token already appears in teacher top-k.
        token_set_ids = torch.cat([teacher_topk_ids, sampled_token_ids_col], dim=-1)
        token_set_mask = torch.cat([torch.ones_like(teacher_topk_ids, dtype=torch.bool), sampled_not_in_topk], dim=-1)

        # Gather student/teacher probabilities on A'.
        student_token_set_log_probs = torch.gather(student_log_probs, dim=-1, index=token_set_ids)
        teacher_token_set_log_probs = torch.cat([teacher_topk_log_probs, sampled_teacher_log_probs], dim=-1)
        student_token_set_probs = student_token_set_log_probs.exp() * token_set_mask
        teacher_token_set_probs = teacher_token_set_log_probs.exp() * token_set_mask

        # Build A'' = A' union {other}; other collects all probability mass outside A'.
        student_other_probs = (1.0 - student_token_set_probs.sum(dim=-1, keepdim=True)).clamp_min(0.0)
        teacher_other_probs = (1.0 - teacher_token_set_probs.sum(dim=-1, keepdim=True)).clamp_min(0.0)

        student_reduced_probs = torch.cat([student_token_set_probs, student_other_probs], dim=-1)
        teacher_reduced_probs = torch.cat([teacher_token_set_probs, teacher_other_probs], dim=-1)
        student_reduced_log_probs = student_reduced_probs.clamp_min(1e-12).log()
        teacher_reduced_log_probs = teacher_reduced_probs.clamp_min(1e-12).log()
        # KL(student || teacher) on the reduced distribution over A''.
        token_kl = student_reduced_probs * (student_reduced_log_probs - teacher_reduced_log_probs)
        distillation_losses = token_kl.sum(dim=-1)
    elif loss_type == "kltopk_approx_mse": 
        sampled_token_ids_col = sampled_token_ids.unsqueeze(-1)
        sampled_not_in_topk = ~(teacher_topk_ids == sampled_token_ids_col).any(dim=-1, keepdim=True)

        # Build A' = teacher top-k tokens union sampled token.
        # The sampled-token slot is kept for a static tensor shape, and masked out
        # when the sampled token already appears in teacher top-k.
        token_set_ids = torch.cat([teacher_topk_ids, sampled_token_ids_col], dim=-1)
        token_set_mask = torch.cat([torch.ones_like(teacher_topk_ids, dtype=torch.bool), sampled_not_in_topk], dim=-1)

        # Gather student/teacher probabilities on A'.
        student_token_set_log_probs = torch.gather(student_log_probs, dim=-1, index=token_set_ids)
        teacher_token_set_log_probs = torch.cat([teacher_topk_log_probs, sampled_teacher_log_probs], dim=-1)
        student_token_set_probs = student_token_set_log_probs.exp() * token_set_mask
        teacher_token_set_probs = teacher_token_set_log_probs.exp() * token_set_mask

        token_kl = student_token_set_probs * (student_token_set_log_probs - teacher_token_set_log_probs) * token_set_mask
        topk_mass_loss_coef = 1.0
        mass_loss = (student_token_set_probs.sum(dim=-1) - teacher_token_set_probs.sum(dim=-1)).pow(2)
        distillation_losses = token_kl.sum(dim=-1) + topk_mass_loss_coef * mass_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_count = overlap_mask.sum(dim=-1)
    overlap_token_advantage_sum = (-overlap_token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )
    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "accumulated_count": accumulated_count,
        "overlap_token_advantage": overlap_token_advantage,
        "debug_teacher_topk_ids": teacher_topk_ids,
        "debug_student_topk_probs": debug_student_topk_probs,
        "debug_teacher_topk_probs": debug_teacher_topk_probs,
    }
