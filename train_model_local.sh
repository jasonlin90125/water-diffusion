# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate water-diffusion
# Set CUDA device to use only one or two GPUs
export CUDA_VISIBLE_DEVICES=2,3

# Remove the existing log file if it exists
rm -f water_diffusion.log

python main_water.py \
    --n_epochs 100 \
    --exp_name edm_water \
    --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 \
    --diffusion_loss_type l2 \
    --batch_size 128 \
    --nf 256 \
    --n_layers 9 \
    --n_report_steps 100 \
    --lr 1e-4 \
    --normalize_factors [1,4,10] \
    --test_epochs 20 \
    --ema_decay 0.9999 \
    --no_wandb \
    >> water_diffusion.log 2>&1