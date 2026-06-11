import os
from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# 1. Config
# =========================

@dataclass(frozen=True)
class ToyConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes: int = 200
    num_samples: int = 8
    epochs: int = 16000
    lr: float = 1e-3
    seed: int = 42
    topk: int = 16
    save_dir: str = "./opd_toy_results"

    # Choose one:
    #   "sample_opd"                         : sampled-token OPD
    #   "direct_kl"                          : exact full-distribution KL(pi || teacher)
    #   "student_topk_opd"                   : Student Top-K OPD, normalized on subset
    #   "teacher_topk_opd"                   : Teacher Top-K OPD, normalized on subset
    #   "teacher_topk_mass_opd"              : Teacher Top-K OPD + top-k mass regularization
    #   "student_topk_unnorm_opd"            : Student Top-K OPD, unnormalized subset KL
    #   "teacher_topk_unnorm_opd"            : Teacher Top-K OPD, unnormalized subset KL
    #   "teacher_topk_unnorm_mass_opd"       : Teacher Top-K unnormalized subset KL + top-k mass regularization
    #   "teacher_topk_ukl_opd"               : Teacher Top-K unnormalized KL with UKL mass correction
    #   "student_dppo_topk_kl_opd"           : DPPO-style collapsed KL, support = student top-k + sampled tokens + other
    #   "teacher_dppo_topk_kl_opd"           : DPPO-style collapsed KL, support = teacher top-k + sampled tokens + other
    #   "teacher_dppo_topk_kl_mass_opd"      : DPPO-style normalized Top-K KL + support mass regularization
    #   "student_dppo_topk_tvd_opd"          : DPPO-style collapsed TVD, support = student top-k + sampled tokens + other
    #   "teacher_dppo_topk_tvd_opd"          : DPPO-style collapsed TVD, support = teacher top-k + sampled tokens + other
    #   "dppo_topk_tv_opd"                   : old DPPO-style behavior top-k + sampled token + other TVD
    method: str = "teacher_dppo_topk_kl_opd"

CONFIG = ToyConfig()
torch.manual_seed(CONFIG.seed)

# =========================
# 2. Distribution builders
# =========================

def gaussian_mixture_logits(
    x,
    mus,
    sigmas,
    amps,
    scale=5.0,
    bias=None,
):
    logits = torch.zeros_like(x)

    for mu, sigma, amp in zip(mus, sigmas, amps):
        logits = logits + amp * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    logits = scale * logits

    if bias is not None:
        logits = logits + bias

    return logits


def normalize_from_logits(logits):
    logprob = torch.log_softmax(logits, dim=-1)
    prob = logprob.exp()
    return prob, logprob

def topk_indices_by_support(student_prob, teacher_prob, k, support):
    if support == "student":
        source_prob = student_prob
    elif support == "teacher":
        source_prob = teacher_prob
    else:
        raise ValueError(f"Unknown support: {support}")

    _, topk_idx = torch.topk(source_prob, k=k, dim=-1)
    return topk_idx

# =========================
# 3. Build toy problem
# =========================

def build_problem(config):
    x = torch.arange(config.num_classes, dtype=torch.float32, device=config.device)

    # Student initialization.
    student_init_logits = gaussian_mixture_logits(
        x,
        mus=[30, 80, 150],
        sigmas=[6, 10, 5],
        amps=[1.0, 1.25, 0.8],
        scale=5.0,
    )

    student_init_prob, student_init_logprob = normalize_from_logits(student_init_logits)

    # Teacher deliberately not too close to student.
    teacher_logits = gaussian_mixture_logits(
        x,
        mus=[48, 112, 172],
        sigmas=[8, 7, 9],
        amps=[0.75, 0.95, 0.8],
        scale=5.5,
    )

    teacher_logits = teacher_logits + 0.25 * torch.sin(x / 13.0)

    teacher_prob, teacher_logprob = normalize_from_logits(teacher_logits)

    return {
        "x": x,
        "student_init_logits": student_init_logits,
        "student_init_prob": student_init_prob,
        "student_init_logprob": student_init_logprob,
        "teacher_logits": teacher_logits,
        "teacher_prob": teacher_prob,
        "teacher_logprob": teacher_logprob,
    }


# =========================
# 4. Plot helper
# =========================

@torch.no_grad()
def plot_distributions(
    x,
    student_init_prob,
    teacher_prob,
    final_student_prob,
    save_name,
    title,
    save_dir,
):
    x_cpu = x.detach().cpu()

    student_init_cpu = student_init_prob.detach().cpu()
    teacher_cpu = teacher_prob.detach().cpu()
    final_cpu = final_student_prob.detach().cpu()

    plt.figure(figsize=(11.5, 5.8), dpi=160)
    ax = plt.gca()
    ax.set_facecolor("white")

    # Teacher fill area.
    plt.fill_between(x_cpu, teacher_cpu, color="#F97316", alpha=0.10, linewidth=0)

    # Initial student: dashed gray.
    plt.plot(x_cpu, student_init_cpu, label="Initial student", color="#64748B", linewidth=2.7, linestyle=(0, (5, 3)), alpha=0.95)

    # Teacher: orange.
    plt.plot(x_cpu, teacher_cpu, label="Teacher", color="#EA580C", linewidth=3.4, linestyle="-", alpha=0.98)

    # Final student: blue.
    plt.plot(x_cpu, final_cpu, label="Final student", color="#2563EB", linewidth=3.2, linestyle="-", alpha=0.98)

    plt.xlabel("Class index", fontsize=13)
    plt.ylabel("Probability", fontsize=13)
    plt.title(title, fontsize=15, pad=14, weight="semibold")

    plt.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.22, color="#94A3B8")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", labelsize=11, colors="#334155")

    ymax = max(teacher_cpu.max().item(), student_init_cpu.max().item(), final_cpu.max().item())
    plt.ylim(0, ymax * 1.18)

    plt.legend(frameon=True, fontsize=11, loc="upper left", framealpha=0.96, edgecolor="#E2E8F0", facecolor="white")

    plt.tight_layout()

    path = os.path.join(save_dir, save_name)
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {path}")


# =========================
# 5. Metrics
# =========================

@torch.no_grad()
def compute_metrics(student_logits, teacher_prob, teacher_logprob):
    student_logprob = torch.log_softmax(student_logits, dim=-1)
    student_prob = student_logprob.exp()

    kl_student_to_teacher = torch.sum(student_prob * (student_logprob - teacher_logprob))

    kl_teacher_to_student = torch.sum(teacher_prob * (teacher_logprob - student_logprob))

    entropy = -(student_prob * student_logprob).sum()

    tvd = 0.5 * torch.sum(torch.abs(student_prob - teacher_prob))

    return {
        "kl_pi_t": kl_student_to_teacher.item(),
        "kl_t_pi": kl_teacher_to_student.item(),
        "entropy": entropy.item(),
        "tvd": tvd.item(),
    }


# =========================
# 6. Method 1: sampled-token OPD
# =========================

def sampled_token_opd_loss(
    train_logits,
    teacher_logprob,
    num_samples,
):
    logprob = torch.log_softmax(train_logits, dim=-1)
    prob = logprob.exp()

    samples = torch.multinomial(prob, num_samples=num_samples, replacement=True)

    sample_logprob = logprob[samples]
    sample_teacher_logprob = teacher_logprob[samples]

    # Reverse-KL OPD reward:
    # R = log q - log p + 1
    reward = sample_teacher_logprob - sample_logprob + 1.0
    advantage = reward.detach()

    # Maximize E[R log p], so minimize negative.
    loss = -(advantage * sample_logprob).mean()

    return loss


# =========================
# 7. Method 2: direct full-distribution KL
# =========================

def direct_kl_loss(
    train_logits,
    teacher_logprob,
):
    logprob = torch.log_softmax(train_logits, dim=-1)
    prob = logprob.exp()

    loss = torch.sum(prob * (logprob - teacher_logprob))

    return loss


# =========================
# 8. Normalized subset Top-K OPD
# =========================

def subset_topk_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    support="student",
    eps=1e-12,
):
    """
    Normalized subset Top-K OPD.

    support = "student":
        S = TopK(pi_theta, k)

    support = "teacher":
        S = TopK(pi_teacher, k)

    Then renormalize both distributions on S:

        p_bar(v) = p(v) / sum_{u in S} p(u)
        q_bar(v) = q(v) / sum_{u in S} q(u)

    Loss:
        KL(p_bar || q_bar)
    """
    logprob = torch.log_softmax(train_logits, dim=-1)
    prob = logprob.exp()
    teacher_prob = teacher_logprob.exp()

    topk_idx = topk_indices_by_support(student_prob=prob, teacher_prob=teacher_prob, k=k, support=support)

    topk_prob = prob[topk_idx]
    topk_teacher_prob = teacher_prob[topk_idx]

    student_mass_on_s = topk_prob.sum().clamp_min(eps)
    teacher_mass_on_s = topk_teacher_prob.sum().clamp_min(eps)

    p_bar = topk_prob / student_mass_on_s
    q_bar = topk_teacher_prob / teacher_mass_on_s

    log_p_bar = torch.log(p_bar.clamp_min(eps))
    log_q_bar = torch.log(q_bar.clamp_min(eps))

    loss = torch.sum(p_bar * (log_p_bar - log_q_bar))

    return loss


def subset_topk_mass_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    support="teacher",
    topk_mass_loss_coef=1.0,
    eps=1e-12,
):
    """
    Normalized subset Top-K OPD with top-k mass regularization.

    The KL term matches the relative distribution on S, while the MSE term
    matches how much total probability student and teacher assign to S.
    """
    logprob = torch.log_softmax(train_logits, dim=-1)
    prob = logprob.exp()
    teacher_prob = teacher_logprob.exp()

    topk_idx = topk_indices_by_support(student_prob=prob, teacher_prob=teacher_prob, k=k, support=support)

    topk_prob = prob[topk_idx]
    topk_teacher_prob = teacher_prob[topk_idx]

    student_mass_on_s = topk_prob.sum().clamp_min(eps)
    teacher_mass_on_s = topk_teacher_prob.sum().clamp_min(eps)

    p_bar = topk_prob / student_mass_on_s
    q_bar = topk_teacher_prob / teacher_mass_on_s

    log_p_bar = torch.log(p_bar.clamp_min(eps))
    log_q_bar = torch.log(q_bar.clamp_min(eps))

    token_kl = torch.sum(p_bar * (log_p_bar - log_q_bar))
    mass_loss = (student_mass_on_s - teacher_mass_on_s).pow(2)
    loss = token_kl + topk_mass_loss_coef * mass_loss

    return loss


# =========================
# 9. Unnormalized subset Top-K OPD
# =========================

def subset_topk_unnorm_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    support="student",
):
    """
    Unnormalized subset Top-K OPD.

    This does NOT renormalize on the selected subset S.

    support = "student":
        S = TopK(pi_theta, k)

    support = "teacher":
        S = TopK(pi_teacher, k)

    Loss:
        sum_{v in S} p(v) * [log p(v) - log q(v)]

    This is a truncated reverse-KL contribution.
    """
    logprob = torch.log_softmax(train_logits, dim=-1)
    prob = logprob.exp()
    teacher_prob = teacher_logprob.exp()

    topk_idx = topk_indices_by_support(student_prob=prob, teacher_prob=teacher_prob, k=k, support=support)

    topk_prob = prob[topk_idx]
    topk_logprob = logprob[topk_idx]
    topk_teacher_logprob = teacher_logprob[topk_idx]

    loss = torch.sum(topk_prob * (topk_logprob - topk_teacher_logprob))

    return loss


def subset_topk_unnorm_mass_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    support="teacher",
    topk_mass_loss_coef=1.0,
):
    """
    Unnormalized subset Top-K OPD with top-k mass regularization.

    This keeps raw top-k probabilities in the token KL term and adds an MSE
    penalty on the total mass assigned to S.
    """
    student_logprob = torch.log_softmax(train_logits, dim=-1)
    student_prob = student_logprob.exp()
    teacher_prob = teacher_logprob.exp()

    topk_idx = topk_indices_by_support(student_prob=student_prob, teacher_prob=teacher_prob, k=k, support=support)

    student_topk_prob = student_prob[topk_idx]
    student_topk_logprob = student_logprob[topk_idx]
    teacher_topk_prob = teacher_prob[topk_idx]
    teacher_topk_logprob = teacher_logprob[topk_idx]

    token_kl = torch.sum(student_topk_prob * (student_topk_logprob - teacher_topk_logprob))

    student_mass_on_s = student_topk_prob.sum()
    teacher_mass_on_s = teacher_topk_prob.sum()
    mass_loss = (student_mass_on_s - teacher_mass_on_s).pow(2)

    loss = token_kl + topk_mass_loss_coef * mass_loss

    return loss


# =========================
# 10. DPPO-style collapsed support
# =========================

def dppo_collapsed_support_indices(
    current_prob,
    teacher_prob,
    k,
    num_samples,
    support="student",
):
    """
    DPPO-style collapsed support.

    support = "student":
        S = TopK(current student, k) union sampled tokens

    support = "teacher":
        S = TopK(teacher, k) union sampled tokens

    sampled tokens are drawn from current student, because in policy learning
    the sampled action/token comes from the current rollout policy.

    Then the rest of vocabulary is collapsed into one "other" bucket.
    """
    topk_idx = topk_indices_by_support(student_prob=current_prob.detach(), teacher_prob=teacher_prob.detach(), k=k, support=support)

    sampled_idx = torch.multinomial(current_prob.detach(), num_samples=num_samples, replacement=True)

    support_idx = torch.unique(torch.cat([topk_idx, sampled_idx], dim=0))

    return support_idx


def append_other_bucket(prob, support_idx, eps=1e-12):
    support_prob = prob[support_idx]
    other_prob = (1.0 - support_prob.sum()).clamp_min(eps)

    reduced_prob = torch.cat([support_prob, other_prob.view(1)], dim=0)

    return reduced_prob, support_prob, other_prob


def dppo_reduced_distributions(
    current_prob,
    teacher_prob,
    support_idx,
    eps=1e-12,
):
    current_reduced, current_s_prob, current_other_prob = append_other_bucket(prob=current_prob, support_idx=support_idx, eps=eps)
    teacher_reduced, teacher_s_prob, teacher_other_prob = append_other_bucket(prob=teacher_prob, support_idx=support_idx, eps=eps)

    return {
        "current_reduced": current_reduced,
        "current_s_prob": current_s_prob,
        "current_other_prob": current_other_prob,
        "teacher_reduced": teacher_reduced,
        "teacher_s_prob": teacher_s_prob,
        "teacher_other_prob": teacher_other_prob,
    }


# =========================
# 11. DPPO-style collapsed Top-K KL OPD
# =========================

def dppo_topk_kl_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    num_samples,
    support="student",
    eps=1e-12,
):
    """
    DPPO-style collapsed KL used as OPD objective.

    It approximates:

        KL(pi_theta || teacher)

    on a reduced distribution:

        [probabilities on S, probability on other]

    where:

        student version:
            S = TopK(pi_theta, k) union sampled tokens

        teacher version:
            S = TopK(teacher, k) union sampled tokens

    The "other" bucket is:

        other = V \\ S
    """
    current_logprob = torch.log_softmax(train_logits, dim=-1)
    current_prob = current_logprob.exp()

    teacher_prob = teacher_logprob.exp()

    support_idx = dppo_collapsed_support_indices(
        current_prob=current_prob, teacher_prob=teacher_prob, k=k, num_samples=num_samples, support=support
    )

    reduced = dppo_reduced_distributions(current_prob=current_prob, teacher_prob=teacher_prob, support_idx=support_idx, eps=eps)

    topk_kl = torch.sum(reduced["current_s_prob"] * (current_logprob[support_idx] - teacher_logprob[support_idx]))

    other_kl = reduced["current_other_prob"] * (torch.log(reduced["current_other_prob"]) - torch.log(reduced["teacher_other_prob"]))

    loss = topk_kl + other_kl

    return loss


def dppo_topk_kl_mass_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    num_samples,
    support="student",
    topk_mass_loss_coef=0.5,
    eps=1e-12,
):
    """
    DPPO-style normalized Top-K KL with support-mass regularization.

    Support S is selected the same way as dppo_topk_kl_opd_loss, but the token
    KL is computed after renormalizing both distributions on S. The mass term
    then explicitly matches how much probability each distribution assigns to S.
    """
    student_logprob = torch.log_softmax(train_logits, dim=-1)
    student_prob = student_logprob.exp()

    teacher_prob = teacher_logprob.exp()

    support_idx = dppo_collapsed_support_indices(
        current_prob=student_prob, teacher_prob=teacher_prob, k=k, num_samples=num_samples, support=support
    )

    student_s_logprob = student_logprob[support_idx]
    teacher_s_logprob = teacher_logprob[support_idx]

    student_s_logprob = student_s_logprob - torch.logsumexp(student_s_logprob, dim=-1)
    teacher_s_logprob = teacher_s_logprob - torch.logsumexp(teacher_s_logprob, dim=-1)

    student_s_prob = student_s_logprob.exp()
    token_kl = torch.sum(student_s_prob * (student_s_logprob - teacher_s_logprob))

    student_mass = student_prob[support_idx].sum().clamp_min(eps)
    teacher_mass = teacher_prob[support_idx].sum().clamp_min(eps)
    mass_loss = (student_mass - teacher_mass).pow(2)

    loss = token_kl + topk_mass_loss_coef * mass_loss

    return loss


def subset_topk_ukl_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    support="teacher",
):
    """
    Unnormalized subset Top-K OPD with UKL mass correction.

    UKL(p, q) = sum_i p_i * (log p_i - log q_i) - p_i + q_i.
    """
    student_logprob = torch.log_softmax(train_logits, dim=-1)
    student_prob = student_logprob.exp()

    teacher_prob = teacher_logprob.exp()

    topk_idx = topk_indices_by_support(student_prob=student_prob, teacher_prob=teacher_prob, k=k, support=support)

    student_topk_logprob = student_logprob[topk_idx]
    teacher_topk_logprob = teacher_logprob[topk_idx]
    student_topk_prob = student_topk_logprob.exp()
    teacher_topk_prob = teacher_topk_logprob.exp()

    token_ukl = (
        student_topk_prob * (student_topk_logprob - teacher_topk_logprob)
        - student_topk_prob
        + teacher_topk_prob
    )
    loss = token_ukl.sum()

    return loss

# =========================
# 12. DPPO-style collapsed Top-K TVD OPD
# =========================

def dppo_topk_tvd_opd_loss(
    train_logits,
    teacher_logprob,
    k,
    num_samples,
    support="student",
    eps=1e-12,
):
    """
    DPPO-style collapsed TVD objective.

    It minimizes TVD between:

        current student reduced distribution
        teacher reduced distribution

    on:

        [probabilities on S, probability on other]

    where:

        student version:
            S = TopK(pi_theta, k) union sampled tokens

        teacher version:
            S = TopK(teacher, k) union sampled tokens

    TVD:

        TVD(p, q) = 0.5 * sum_i |p_i - q_i|

    Note:
        This is not reverse-KL OPD mathematically.
        It is a DPPO-style collapsed-support distribution matching loss.
    """
    current_logprob = torch.log_softmax(train_logits, dim=-1)
    current_prob = current_logprob.exp()

    teacher_prob = teacher_logprob.exp()

    support_idx = dppo_collapsed_support_indices(
        current_prob=current_prob, teacher_prob=teacher_prob, k=k, num_samples=num_samples, support=support
    )

    reduced = dppo_reduced_distributions(current_prob=current_prob, teacher_prob=teacher_prob, support_idx=support_idx, eps=eps)

    loss = 0.5 * torch.sum(torch.abs(reduced["current_reduced"] - reduced["teacher_reduced"]))

    return loss


# =========================
# 13. Old behavior-support DPPO TVD
# =========================

def dppo_behavior_support_indices(
    behavior_logprob,
    k,
    num_samples,
):
    """
    Original DPPO-style behavior support:

        S = TopK(mu, k) union sampled tokens

    In DPPO:
        mu = behavior / rollout policy.

    In this toy code:
        mu = initial student distribution.
    """
    behavior_prob = behavior_logprob.exp()

    _, behavior_topk_idx = torch.topk(behavior_prob, k=k, dim=-1)

    sampled_idx = torch.multinomial(behavior_prob, num_samples=num_samples, replacement=True)

    support_idx = torch.unique(torch.cat([behavior_topk_idx, sampled_idx], dim=0))

    return support_idx


def dppo_topk_tv_opd_loss(
    train_logits,
    behavior_logprob,
    teacher_logprob,
    k,
    num_samples,
    eps=1e-12,
):
    """
    Old behavior-support DPPO-style TVD.

    Support:
        S = TopK(initial student / behavior, k) union sampled tokens

    This is kept for comparison with the new student/teacher TVD versions.
    """
    current_logprob = torch.log_softmax(train_logits, dim=-1)
    current_prob = current_logprob.exp()

    teacher_prob = teacher_logprob.exp()

    support_idx = dppo_behavior_support_indices(behavior_logprob=behavior_logprob, k=k, num_samples=num_samples)

    current_reduced, _, _ = append_other_bucket(prob=current_prob, support_idx=support_idx, eps=eps)
    teacher_reduced, _, _ = append_other_bucket(prob=teacher_prob, support_idx=support_idx, eps=eps)

    loss = 0.5 * torch.sum(torch.abs(current_reduced - teacher_reduced))

    return loss


# =========================
# 14. Method registry and training
# =========================

METHODS = {
    "sample_opd": {
        "title": "Sampled-token OPD",
        "loss_fn": sampled_token_opd_loss,
        "uses_num_samples": True,
    },
    "direct_kl": {
        "title": "Direct KL Minimization",
        "loss_fn": direct_kl_loss,
    },
    "student_topk_opd": {
        "title": "Student Top-{topk} OPD",
        "loss_fn": subset_topk_opd_loss,
        "support": "student",
        "uses_topk": True,
    },
    "teacher_topk_opd": {
        "title": "Teacher Top-{topk} OPD",
        "loss_fn": subset_topk_opd_loss,
        "support": "teacher",
        "uses_topk": True,
    },
    "teacher_topk_mass_opd": {
        "title": "Teacher Top-{topk} OPD + Mass",
        "loss_fn": subset_topk_mass_opd_loss,
        "support": "teacher",
        "uses_topk": True,
    },
    "student_topk_unnorm_opd": {
        "title": "Student Top-{topk} Unnormalized OPD",
        "loss_fn": subset_topk_unnorm_opd_loss,
        "support": "student",
        "uses_topk": True,
    },
    "teacher_topk_unnorm_opd": {
        "title": "Teacher Top-{topk} Unnormalized OPD",
        "loss_fn": subset_topk_unnorm_opd_loss,
        "support": "teacher",
        "uses_topk": True,
    },
    "teacher_topk_unnorm_mass_opd": {
        "title": "Teacher Top-{topk} Unnormalized OPD + Mass",
        "loss_fn": subset_topk_unnorm_mass_opd_loss,
        "support": "teacher",
        "uses_topk": True,
    },
    "student_dppo_topk_kl_opd": {
        "title": "Student DPPO-style Top-{topk} Collapsed KL OPD",
        "loss_fn": dppo_topk_kl_opd_loss,
        "support": "student",
        "uses_topk": True,
        "uses_num_samples": True,
    },
    "teacher_dppo_topk_kl_opd": {
        "title": "Teacher DPPO-style Top-{topk} Collapsed KL OPD",
        "loss_fn": dppo_topk_kl_opd_loss,
        "support": "teacher",
        "uses_topk": True,
        "uses_num_samples": True,
    },
    "teacher_dppo_topk_kl_mass_opd": {
        "title": "Teacher DPPO-style Top-{topk} KL + Mass OPD",
        "loss_fn": dppo_topk_kl_mass_opd_loss,
        "support": "teacher",
        "uses_topk": True,
        "uses_num_samples": True,
    },
    "teacher_topk_ukl_opd": {
        "title": "Teacher Top-{topk} UKL OPD",
        "loss_fn": subset_topk_ukl_opd_loss,
        "support": "teacher",
        "uses_topk": True,
    },
    "student_dppo_topk_tvd_opd": {
        "title": "Student DPPO-style Top-{topk} Collapsed TVD OPD",
        "loss_fn": dppo_topk_tvd_opd_loss,
        "support": "student",
        "uses_topk": True,
        "uses_num_samples": True,
    },
    "teacher_dppo_topk_tvd_opd": {
        "title": "Teacher DPPO-style Top-{topk} Collapsed TVD OPD",
        "loss_fn": dppo_topk_tvd_opd_loss,
        "support": "teacher",
        "uses_topk": True,
        "uses_num_samples": True,
    },
    "dppo_topk_tv_opd": {
        "title": "Old Behavior DPPO-style Top-{topk} Collapsed TVD OPD",
        "loss_fn": dppo_topk_tv_opd_loss,
        "uses_behavior_logprob": True,
        "uses_topk": True,
        "uses_num_samples": True,
    },
    
    
}


def available_methods():
    return ", ".join(METHODS)


def method_spec(method_name):
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Choose from: {available_methods()}")

    return METHODS[method_name]


def method_title(method_name, config):
    return method_spec(method_name)["title"].format(topk=config.topk)


def compute_training_loss(
    method_name,
    train_logits,
    student_init_logprob,
    teacher_logprob,
    config,
):
    spec = method_spec(method_name)
    loss_kwargs = {
        "train_logits": train_logits,
        "teacher_logprob": teacher_logprob,
    }

    if spec.get("uses_behavior_logprob"):
        loss_kwargs["behavior_logprob"] = student_init_logprob
    if spec.get("uses_topk"):
        loss_kwargs["k"] = config.topk
    if spec.get("uses_num_samples"):
        loss_kwargs["num_samples"] = config.num_samples
    if "support" in spec:
        loss_kwargs["support"] = spec["support"]

    return spec["loss_fn"](**loss_kwargs)


def format_metrics_for_progress(loss, metrics):
    return {
        "loss": f"{loss.item():.4f}",
        "KL(pi||T)": f"{metrics['kl_pi_t']:.4f}",
        "KL(T||pi)": f"{metrics['kl_t_pi']:.4f}",
        "TVD": f"{metrics['tvd']:.4f}",
        "H": f"{metrics['entropy']:.3f}",
    }


def train(
    student_init_logits,
    student_init_logprob,
    teacher_prob,
    teacher_logprob,
    method_name,
    config,
):
    title = method_title(method_name, config)

    train_logits = student_init_logits.clone().detach().requires_grad_(True)
    optimizer = torch.optim.AdamW([train_logits], lr=config.lr, weight_decay=0.0)
    pbar = tqdm(range(config.epochs), desc=f"Training {title}")

    for epoch in pbar:
        loss = compute_training_loss(
            method_name=method_name,
            train_logits=train_logits,
            student_init_logprob=student_init_logprob,
            teacher_logprob=teacher_logprob,
            config=config,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == config.epochs - 1:
            metrics = compute_metrics(
                student_logits=train_logits,
                teacher_prob=teacher_prob,
                teacher_logprob=teacher_logprob,
            )
            pbar.set_postfix(format_metrics_for_progress(loss, metrics))

    final_logits = train_logits.detach()
    final_prob = torch.softmax(final_logits, dim=-1)
    return final_prob, final_logits


# =========================
# 15. Main
# =========================

def main():
    config = CONFIG
    os.makedirs(config.save_dir, exist_ok=True)

    problem = build_problem(config)

    x = problem["x"]
    student_init_logits = problem["student_init_logits"]
    student_init_prob = problem["student_init_prob"]
    student_init_logprob = problem["student_init_logprob"]
    teacher_prob = problem["teacher_prob"]
    teacher_logprob = problem["teacher_logprob"]

    final_student_prob, final_logits = train(
        student_init_logits=student_init_logits,
        student_init_logprob=student_init_logprob,
        teacher_prob=teacher_prob,
        teacher_logprob=teacher_logprob,
        method_name=config.method,
        config=config,
    )

    final_metrics = compute_metrics(
        student_logits=final_logits,
        teacher_prob=teacher_prob,
        teacher_logprob=teacher_logprob,
    )

    print("\nFinal diagnostics")
    print("-----------------")
    print(f"Method:                 {config.method}")
    print(f"KL(student || teacher): {final_metrics['kl_pi_t']:.6f}")
    print(f"KL(teacher || student): {final_metrics['kl_t_pi']:.6f}")
    print(f"TVD(student, teacher):  {final_metrics['tvd']:.6f}")
    print(f"Student entropy:        {final_metrics['entropy']:.6f}")

    save_name = f"{config.method}.png"
    title = method_title(config.method, config)

    plot_distributions(
        x=x,
        student_init_prob=student_init_prob,
        teacher_prob=teacher_prob,
        final_student_prob=final_student_prob,
        save_name=save_name,
        title=title,
        save_dir=config.save_dir,
    )


if __name__ == "__main__":
    main()
