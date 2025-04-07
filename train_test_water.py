import wandb
# Make sure utils points to the diffusion utils, not water utils
from equivariant_diffusion import utils as diffusion_utils
import numpy as np
import utils # General utils like Queue
from water import losses # Water specific losses (should be general now)
import time
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

def train_epoch(args, loader, epoch, model, # model is potentially DDP wrapped
                model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, rank, wandb): # Added rank, wandb
    # Use model directly (works for both single and DDP)
    model.train()
    if args.ema_decay > 0: model_ema.train() # Set EMA model mode if used

    nll_epoch = []
    n_iterations = len(loader)

    # Wrap loader in tqdm only on rank 0 for clean progress bar
    if rank == 0:
        pbar = tqdm(enumerate(loader), total=n_iterations)
    else:
        pbar = enumerate(loader)

    for i, batch in pbar: # Use pbar iterator
        # Unpack batch according to the new collate_fn structure
        x = batch['positions'].to(device, dtype)         # [B, N, 3, 3]
        molecule_mask = batch['molecule_mask'].to(device, dtype) # [B, N]
        atom_mask = batch['atom_mask'].to(device, dtype)     # [B, N*3]
        one_hot = batch['one_hot'].to(device, dtype)         # [B, N*3, num_types]
        atom_edge_mask = batch['edge_mask'].to(device, dtype)  # [B, N*3, N*3]

        # Center positions (using molecule mask might be tricky, use atom mask for overall COM=0)
        x_flat = x.view(x.shape[0], -1, 3) # Flatten to [B, N*3, 3] for centering
        atom_mask_unsqueeze = atom_mask.unsqueeze(-1) # [B, N*3, 1]
        x_centered_flat = diffusion_utils.remove_mean_with_mask(x_flat, atom_mask_unsqueeze)
        x = x_centered_flat.view(x.shape) # Reshape back to [B, N, 3, 3]

        # Features 'h' are derived from one_hot
        h = {'categorical': one_hot, 'integer': torch.zeros(0).to(device)} # No integer features

        context = None # No context implemented here

        optim.zero_grad()

        # === Call Diffusion Model Forward ===
        # Pass atom_mask as node_mask, atom_edge_mask as edge_mask
        nll = model(x, h, node_mask=atom_mask, edge_mask=atom_edge_mask, context=context)
        # nll is already averaged over batch inside the model's forward

        # Backpropagation
        loss = nll # The model's forward returns the final loss scalar
        loss.backward()

        # Gradient Clipping
        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.
        optim.step()

        # Update EMA
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model.module if isinstance(model, DDP) else model)

        if rank == 0: # Log only from rank 0
            log_msg = f"Epoch: {epoch}, iter: {i}/{n_iterations}, Loss {loss.item():.4f}, GradNorm: {grad_norm:.1f}"
            # Update tqdm description if used
            if isinstance(pbar, tqdm):
                pbar.set_description(log_msg)
            elif i % args.n_report_steps == 0:
                print(log_msg) # Fallback print

            if wandb:
                wandb.log({"Batch NLL": loss.item(), "Gradient Norm": grad_norm}, commit=True)

        if args.break_train_epoch:
            break

    if rank == 0:
        epoch_nll_avg = np.mean(nll_epoch) if nll_epoch else 0
        if wandb:
            wandb.log({"Train Epoch NLL": epoch_nll_avg}, commit=False) # Commit handled by validation step


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'): # Removed nodes_dist
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0.0
        n_samples = 0 # Count samples correctly

        n_iterations = len(loader)
        for i, batch in enumerate(loader):
            # Unpack batch
            x = batch['positions'].to(device, dtype)         # [B, N, 3, 3]
            molecule_mask = batch['molecule_mask'].to(device, dtype) # [B, N]
            atom_mask = batch['atom_mask'].to(device, dtype)     # [B, N*3]
            one_hot = batch['one_hot'].to(device, dtype)         # [B, N*3, num_types]
            atom_edge_mask = batch['edge_mask'].to(device, dtype)  # [B, N*3, N*3]

            batch_size = x.shape[0]

            # Center positions
            x_flat = x.view(x.shape[0], -1, 3)
            atom_mask_unsqueeze = atom_mask.unsqueeze(-1)
            x_centered_flat = diffusion_utils.remove_mean_with_mask(x_flat, atom_mask_unsqueeze)
            x = x_centered_flat.view(x.shape)

            h = {'categorical': one_hot, 'integer': torch.zeros(0).to(device)}
            context = None

            # === Call Diffusion Model Forward (calculates NLL directly) ===
            nll = eval_model(x, h, node_mask=atom_mask, edge_mask=atom_edge_mask, context=context)

            nll_epoch += nll.item() * batch_size # Accumulate total NLL for the epoch
            n_samples += batch_size

            if i % args.n_report_steps == 0:
                 current_avg_nll = nll_epoch / n_samples if n_samples > 0 else 0
                 print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                       f"Avg NLL: {current_avg_nll:.4f}")

    avg_nll_epoch = nll_epoch / n_samples if n_samples > 0 else 0
    print(f"\n{partition} Epoch {epoch} Average NLL: {avg_nll_epoch:.4f}")
    return avg_nll_epoch