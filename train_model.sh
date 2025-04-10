#!/bin/bash
#SBATCH --job-name=edm_train
#SBATCH --output=edm_train.out
#SBATCH --error=edm_train.err
#SBATCH -p volta-gpu
#SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=shuhang@unc.edu

# Activate conda environment
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate diffusion

module load cuda/12.5
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi --query-gpu=memory.total,memory.used --format=csv

python main_water.py \
    --n_epochs 1000 \
    --exp_name edm_water \
    --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 9 \
    --lr 1e-4 \
    --n_report_steps 100 \
    --normalize_factors [1,4,10] \
    --test_epochs 20 \
    --ema_decay 0.9999
