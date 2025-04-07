# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate water-diffusion # Your environment name

# Set CUDA device visibility *before* torchrun
# This tells torchrun which GPUs are available to be assigned ranks 0, 1, 2...
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define the number of GPUs to use based on CUDA_VISIBLE_DEVICES
# This is a simple way to count GPUs listed in the variable
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Found $NUM_GPUS GPUs specified in CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Launching $NUM_GPUS processes for DDP training..."

# Remove the existing log file if it exists
LOG_FILE="water_diffusion_ddp_local.log" # Changed log file name
rm -f $LOG_FILE
echo "Starting DDP training, logging to $LOG_FILE"

# --- Launch with torchrun ---
# --standalone: For single-node training
# --nnodes=1: Explicitly state single-node
# --nproc_per_node=$NUM_GPUS: Launch one process per visible GPU
export NCCL_DEBUG=INFO # Optional: for debugging NCCL communication
export PYTHONFAULTHANDLER=1 # Optional: better tracebacks on crash

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main_water.py \
    --n_epochs 200 \
    --exp_name edm_water \
    --n_stability_samples 1000 \
    --aggregation_method mean \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 \
    --diffusion_loss_type l2 \
    --batch_size 32 \
    --nf 256 \
    --n_layers 9 \
    --n_report_steps 100 \
    --lr 1e-4 \
    --normalize_factors "[5,1,1]" \
    --test_epochs 20 \
    --ema_decay 0.9999 \
    --no_wandb \
    >> $LOG_FILE 2>&1 &

# Wait for background process to finish (optional, useful in scripts)
wait
echo "Local DDP training finished. Check $LOG_FILE for logs."

# --- End Launch with torchrun ---