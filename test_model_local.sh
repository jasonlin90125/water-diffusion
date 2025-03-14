# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate water-diffusion

# Set CUDA device to use only one or two GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Remove the existing log file if it exists
rm -f generate_molecules.log

python generate_molecules.py \
    --exp_name edm_water \
    --ema \
    --n_samples 100 \
    --batch_size 10 \
    --nf 256 \
    --n_layers 9 \
    >> generate_molecules.log 2>&1