# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import os, copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from water import dataset
from water.models import get_optim, get_model
#from water.utils import prepare_context, compute_mean_mad
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import torch.distributed as dist # Add distributed import
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP # Import DDP
from torch.utils.data.distributed import DistributedSampler # Import Sampler
import time
import pickle
from train_test_water import train_epoch, test #, analyze_and_save

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='water',
                    help='water | qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--dataset_path', type=str, default='data/water/water.h5',
                    help='Path to h5 file containing water positions.')
parser.add_argument('--datadir', type=str, default='water/temp',
                    help='water data directory')
parser.add_argument('--radius', type=float, default=0.5,
                    help='Radius of the cutoff sphere in nanometers.')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, # True -> False
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')

def setup_ddp():
    """Initializes the distributed environment."""
    # These environment variables are typically set by torchrun/torch.distributed.launch
    rank = int(os.environ.get("RANK", "0")) # Global rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) # Rank on the current node
    world_size = int(os.environ.get("WORLD_SIZE", "1")) # Total number of processes

    if world_size > 1:
        print(f"Initializing DDP: Rank {rank}, Local Rank {local_rank}, World Size {world_size}")
        # NCCL is the recommended backend for NVIDIA GPUs
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank) # Set the device for this specific process
        is_distributed = True
        print(f"Rank {rank} process initialized on cuda:{local_rank}")
    else:
        print("Running in single-process mode (no DDP initialization).")
        is_distributed = False
        local_rank = 0 # Default to device 0 if not distributed

    # Ensure device consistency
    device = torch.device(f"cuda:{local_rank}")

    return rank, local_rank, world_size, device, is_distributed

def cleanup_ddp():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
    print("DDP Cleanup finished.")

def main():
    args = parser.parse_args()

    # === DDP Setup ===
    # Call setup early to get rank, device, etc.
    rank, local_rank, world_size, device, is_distributed = setup_ddp()
    # Override args.cuda and set device based on DDP setup
    args.cuda = torch.cuda.is_available()
    args.device = device
    # --- End DDP Setup ---

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    atom_encoder = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']

    # args, unparsed_args = parser.parse_known_args()
    args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

    dtype = torch.float32

    if args.resume is not None:
        exp_name = args.exp_name + '_resume'
        start_epoch = args.start_epoch
        resume = args.resume
        wandb_usr = args.wandb_usr
        normalization_factor = args.normalization_factor
        aggregation_method = args.aggregation_method

        with open(join(args.resume, 'args.pickle'), 'rb') as f:
            args = pickle.load(f)

        args.resume = resume
        args.break_train_epoch = False

        args.exp_name = exp_name
        args.start_epoch = start_epoch
        args.wandb_usr = wandb_usr

        # Careful with this -->
        if not hasattr(args, 'normalization_factor'):
            args.normalization_factor = normalization_factor
        if not hasattr(args, 'aggregation_method'):
            args.aggregation_method = aggregation_method

        print(args)

    if rank == 0: # Only log/create folders from the main process
        utils.create_folders(args)
        print(args)

    # === Wandb Setup ===
    if rank == 0 and not args.no_wandb: # Only init WandB on rank 0
        mode = 'online' if args.online else 'offline'
        kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion_water', 'config': args, # Changed project name
                  'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')
    elif args.no_wandb:
        wandb = None # Ensure wandb is None if disabled
    else: # Other ranks don't use wandb fully
        wandb = None # Or use a dummy object if wandb calls are scattered
    # --- End Wandb Setup ---

# === Retrieve Dataloaders ===
    # Load the full datasets first
    # The original retrieve_dataloaders loads dicts for 'train', 'valid', 'test'
    all_dataloaders = dataset.retrieve_dataloaders(args)

    # Create DistributedSampler for the training set if distributed
    train_sampler = None
    if is_distributed:
        # Get the underlying dataset object if it's a Subset
        train_dataset = all_dataloaders['train'].dataset
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        # Need to re-create the DataLoader with the sampler
        # Ensure collate_fn is correctly passed from the original loader setup if needed
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler, # Use the sampler
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  collate_fn=all_dataloaders['train'].collate_fn) # Reuse collate_fn
    else:
        train_loader = all_dataloaders['train'] # Use original loader if not distributed

    # Validation and Test loaders generally don't need distributed sampling,
    # evaluated typically on rank 0 or aggregated across ranks.
    valid_loader = all_dataloaders['valid']
    test_loader = all_dataloaders['test']
    # Re-assign to the main dict structure used later
    dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    context_node_nf = 0
    property_norms = None
    args.context_node_nf = context_node_nf

    # === Create Model ===
    # Pass the correct 'device' for the current process
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    # if prop_dist is not None: prop_dist.set_normalizer(property_norms) # If props used
    model = model.to(device) # Move model to the assigned GPU *before* DDP wrapping
    optim = get_optim(args, model)
    # --- End Create Model ---
    
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)

    # === Resume Logic ===
    if args.resume is not None:
        # Ensure checkpoint loading works correctly in a distributed setting
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} # Map checkpoint saved on GPU 0 to current GPU
        checkpoint = torch.load(join(args.resume, 'model.pt'), map_location=map_location) # Example: save model and optim in one file
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch = checkpoint.get('epoch', 0) + 1 # Resume from next epoch
        print(f"Rank {rank} resumed model from epoch {args.start_epoch-1}")
        # Ensure gradnorm_queue etc. are potentially loaded or reset
    # --- End Resume Logic ---


    # === DDP Model Wrapping ===
    if is_distributed:
        # Wrap the model with DDP *after* moving it to the correct device
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print(f"Rank {rank} wrapped model with DDP.")
    # 'model_dp' is no longer needed, just use 'model' directly
    # --- End DDP Wrapping ---

    # === EMA Setup ===
    # EMA needs careful handling with DDP. The EMA model should *not* be wrapped by DDP.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model.module if is_distributed else model).to(device) # Get underlying model if DDP, move to device
        ema = flow_utils.EMA(args.ema_decay)
        model_ema_dp = model_ema # EMA model runs independently on each rank if needed for eval
    else:
        ema = None
        model_ema = model.module if is_distributed else model # Reference underlying model
        model_ema_dp = model_ema
    # --- End EMA Setup ---

    best_nll_val = 1e8
    best_nll_test = 1e8

    # === Training Loop ===
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        # === Set epoch for DistributedSampler ===
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # --- End Set Epoch ---

        if rank == 0: # Print epoch info only on rank 0
            print(f"Epoch {epoch} / {args.n_epochs} --- Rank {rank}")

        # Pass 'model' (which is DDP wrapped if is_distributed) to train_epoch
        # Inside train_epoch, operations like loss.backward() work correctly with DDP
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, # Pass the (potentially wrapped) model
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    optim=optim, nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, prop_dist=prop_dist, rank=rank, wandb=wandb) # Pass rank and wandb

        epoch_time = time.time() - start_epoch
        if rank == 0: # Log time only on rank 0
            print(f"Epoch took {epoch_time:.1f} seconds.")
            if wandb: wandb.log({"Epoch Time": epoch_time}, commit=False)

        # === Validation/Testing (typically on Rank 0) ===
        if epoch % args.test_epochs == 0:
            # if isinstance(model, en_diffusion.EnVariationalDiffusion): # Access module if DDP
            #     log_info_model = model.module if is_distributed else model
            #     if rank == 0 and wandb: wandb.log(log_info_model.log_info(), commit=True)

            # Perform validation/testing primarily on rank 0 to avoid redundancy
            nll_val = torch.tensor(0.0, device=device) # Placeholder tensor
            nll_test = torch.tensor(0.0, device=device)

            if rank == 0:
                print(f"--- Rank {rank} starting evaluation ---")
                # Evaluate using the EMA model
                current_model_ema = model_ema # Already on the correct device
                nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=current_model_ema, # Pass EMA model
                               partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                               property_norms=property_norms)
                nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=current_model_ema, # Pass EMA model
                                partition='Test', device=device, dtype=dtype,
                                nodes_dist=nodes_dist, property_norms=property_norms)
                print(f"--- Rank {rank} finished evaluation ---")

            # Synchronize results if needed (e.g., broadcast best loss from rank 0)
            # For simplicity here, rank 0 handles decisions and saving
            if is_distributed:
                # Ensure all processes wait here before rank 0 potentially saves model
                dist.barrier()
                # If validation loss needed on all ranks (e.g., for scheduler), broadcast from rank 0
                # dist.broadcast(nll_val, src=0) # Example

            if rank == 0: # Rank 0 handles best model logic and saving
                if nll_val < best_nll_val:
                    best_nll_val = nll_val
                    best_nll_test = nll_test
                    best_epoch = epoch
                    if args.save_model:
                        args.current_epoch = epoch + 1
                        utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                        utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                        if args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                        with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                            pickle.dump(args, f)

                print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
                print('Best val loss: %.4f \t Best test loss:  %.4f \t epoch %d' % (best_nll_val, best_nll_test, best_epoch))
                if wandb:
                    wandb.log({"Val loss ": nll_val}, commit=False)
                    wandb.log({"Test loss ": nll_test}, commit=False)
                    wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True) # Commit here
            # --- End Validation/Testing ---

        # Ensure all processes finish the epoch before starting the next one
        if is_distributed:
            dist.barrier()
    # === End Training Loop ===

    # === Cleanup DDP ===
    cleanup_ddp()
    # --- End Cleanup ---

if __name__ == "__main__":
    main()
