#!/bin/bash
#SBATCH --partition=A100
#SBATCH --gres=gpu:6
#SBATCH --job-name=train          # 作业名称
#SBATCH --nodes=1                        # 使用 1 个节点
#SBATCH --ntasks=1                       # 总任务数
#SBATCH --cpus-per-task=8               # 每任务 4 个 CPU 核
#SBATCH --mem=128G                        # 内存
#SBATCH --time=96:00:00                  # 最长运行时间 30 分钟
#SBATCH --output=eval_test_%j.out    # 标准输出（%j = 作业ID）
#SBATCH --error=eavl_test_%j.err     # 错误输出

# =================== 环境加载 ===================
echo "=== 开始加载环境 ==="
source /data/softwares/miniconda3/26.3.2-2/etc/profile.d/conda.sh
conda activate /data/home/zhanghx/conda/envs/verl

echo "当前 Python: $(which python)"
echo "PyTorch 路径: $(python -c 'import torch; print(torch.__file__)')"

# =================== 执行测试 ===================
echo "=== 开始测试 ==="
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh

#--nodelist=gpu-05