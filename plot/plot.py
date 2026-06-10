import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = "debug_topk_probs/topk4_normalize/rank_0_topk_probs.jsonl"
DEFAULT_OUTPUT = "plot/first12_teacher_student_topk_bar.png"
LAST_OUTPUT = "plot/last12_teacher_student_topk_bar.png"


def read_samples(path, num_samples, last=False):
    samples = []
    bad_lines = []
    tail_samples = deque(maxlen=num_samples) if last else None
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not last and len(samples) >= num_samples:
                break
            if not line.strip():
                continue
            try:
                sample = (line_no, json.loads(line))
            except json.JSONDecodeError as exc:
                bad_lines.append((line_no, str(exc)))
                continue
            if last:
                tail_samples.append(sample)
            else:
                samples.append(sample)
    if last:
        samples = list(tail_samples)
    return samples, bad_lines


def plot_samples(samples, output_path):
    teacher_sums = [0.0, 0.0, 0.0, 0.0]
    student_sums = [0.0, 0.0, 0.0, 0.0]
    token_count_total = 0

    for _, sample in samples:
        teacher_probs = sample["teacher_topk_probs"]
        student_probs = sample["student_topk_probs"]
        token_count = min(len(teacher_probs), len(student_probs))
        token_count_total += token_count

        for token_idx in range(token_count):
            for top_idx in range(4):
                teacher_sums[top_idx] += teacher_probs[token_idx][top_idx]
                student_sums[top_idx] += student_probs[token_idx][top_idx]

    teacher_means = [value / token_count_total for value in teacher_sums]
    student_means = [value / token_count_total for value in student_sums]

    fig, ax = plt.subplots(figsize=(8, 5))
    top_labels = ["top1", "top2", "top3", "top4"]
    x_positions = range(len(top_labels))
    width = 0.36

    ax.bar(
        [x - width / 2 for x in x_positions],
        teacher_means,
        width,
        label="teacher",
        color="tab:blue",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        student_means,
        width,
        label="student",
        color="tab:orange",
    )

    ax.set_xticks(list(x_positions), top_labels)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.set_ylabel("mean probability")
    ax.set_title(f"First {len(samples)} samples: mean top-k probabilities ({token_count_total} tokens)")
    ax.legend(frameon=False)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return token_count_total, teacher_means, student_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--last", action="store_true", help="plot the last N valid samples")
    args = parser.parse_args()

    if args.last and args.output == DEFAULT_OUTPUT:
        args.output = LAST_OUTPUT

    samples, bad_lines = read_samples(args.input, args.num_samples, last=args.last)
    if not samples:
        raise RuntimeError(f"No valid samples found in {args.input}")

    token_count, teacher_means, student_means = plot_samples(samples, Path(args.output))
    print(f"saved {args.output}")
    print(f"plotted {len(samples)} samples")
    print(f"tokens {token_count}")
    print(f"teacher {[round(value, 6) for value in teacher_means]}")
    print(f"student {[round(value, 6) for value in student_means]}")
    if bad_lines:
        print(f"skipped malformed lines: {bad_lines[:5]}")


if __name__ == "__main__":
    main()
