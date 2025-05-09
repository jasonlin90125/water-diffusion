from rdkit import Chem
from rdkit.Chem import AllChem
import os
import torch
import argparse
import numpy as np
import pickle
import utils
from configs.datasets_config import get_dataset_info
from water.models import get_model
from equivariant_diffusion import utils as flow_utils
from water import dataset
from tqdm import tqdm

def load_model_from_checkpoint(args, device, dataset_info, prior_nodes_dist=None):
    """Load a pre-trained model from checkpoint."""
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, None)
    
    if args.ema:
        model_path = 'outputs/%s/generative_model_ema.npy' % args.exp_name
    else:
        model_path = 'outputs/%s/generative_model.npy' % args.exp_name
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, nodes_dist, prop_dist

def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    max_n_nodes = 35
    #max_n_nodes = dataset_info['max_n_nodes']

    #assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    context = None

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

        #assert_correctly_masked(x, node_mask)
        #assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        #assert_correctly_masked(one_hot.float(), node_mask)
        #if args.include_charges:
        #    assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask

def get_args():
    """Parse command line arguments for molecule generation."""
    parser = argparse.ArgumentParser(description='Generate molecules with E3Diffusion model')
    
    # Model loading parameters
    parser.add_argument('--exp_name', type=str, default='debug_10', help='Experiment name to load model from')
    parser.add_argument('--ema', action='store_true', help='Use EMA model weights')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generation parameters
    parser.add_argument('--n_samples', type=int, default=100, help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--fix_atoms', type=int, default=None, help='Fix number of atoms per molecule')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='generated_molecules', help='Directory to save generated molecules')
    
    # Same arguments as the model training to ensure compatibility
    parser.add_argument('--model', type=str, default='egnn_dynamics')
    parser.add_argument('--probabilistic_model', type=str, default='diffusion')
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
    parser.add_argument('--diffusion_loss_type', type=str, default='l2')
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--inv_sublayers', type=int, default=1)
    parser.add_argument('--nf', type=int, default=128)
    parser.add_argument('--tanh', type=eval, default=True)
    parser.add_argument('--attention', type=eval, default=True)
    parser.add_argument('--norm_constant', type=float, default=1)
    parser.add_argument('--sin_embedding', type=eval, default=False)
    parser.add_argument('--dataset', type=str, default='water')
    parser.add_argument('--dataset_path', type=str, default='data/water/water.h5')
    parser.add_argument('--datadir', type=str, default='water/temp')
    parser.add_argument('--radius', type=float, default=0.5)
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--include_charges', type=eval, default=False)
    parser.add_argument('--conditioning', nargs='+', default=[])
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1])
    parser.add_argument('--normalization_factor', type=float, default=1)
    parser.add_argument('--aggregation_method', type=str, default='sum')
    
    args = parser.parse_args()
    return args

def save_oxygen_pdb(xyz_coordinates, output_path):
    """Save a PDB file with only oxygen atoms given XYZ coordinates."""
    with open(output_path, 'w') as f:
        atom_index = 1
        for coord in xyz_coordinates:
            x, y, z = coord
            f.write(f"ATOM  {atom_index:5d}  O   HOH A   1    {x*10:8.3f}{y*10:8.3f}{z*10:8.3f}  1.00  0.00           O\n")
            atom_index += 1

def main():
    # Parse arguments
    args = get_args()

    print(args.exp_name)
    
    # Set device
    device = torch.device(args.device)
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    
    # Try to load original arguments from the experiment to ensure compatibility
    try:
        with open(f'outputs/{args.exp_name}/args.pickle', 'rb') as f:
            train_args = pickle.load(f)
            # Update current args with necessary training parameters
            for key, value in vars(train_args).items():
                if key not in ['n_samples', 'batch_size', 'temperature', 'output_dir', 'device', 'ema']:
                    setattr(args, key, value)
    except FileNotFoundError:
        print(f"Warning: Could not find args.pickle file for experiment {args.exp_name}. Using default arguments.")
    
    # Set context node features
    args.context_node_nf = 0
    
    # Load model
    model, nodes_dist, prop_dist = load_model_from_checkpoint(args, device, dataset_info)
    
    # Generate molecules
    print(f"Generating {args.n_samples} molecules...")
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate molecules in batches
    all_molecules = []
    num_batches = (args.n_samples + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(num_batches)):
        # Determine batch size for the last batch
        current_batch_size = min(args.batch_size, args.n_samples - i * args.batch_size)
        
        # Determine number of atoms per molecule
        if args.fix_atoms is not None:
            nodesxsample = torch.ones(current_batch_size, dtype=torch.int64) * args.fix_atoms
        else:
            #nodesxsample = nodes_dist.sample(current_batch_size)
            nodesxsample = torch.randint(15, 21, (current_batch_size,), dtype=torch.int64)
        
        # Sample molecules
        one_hot, charges, positions, node_mask = sample(
            args, device, model, dataset_info, 
            prop_dist=prop_dist,
            nodesxsample=nodesxsample)

        # Save positions to XYZ file
        for j in range(current_batch_size):
            xyz_coordinates = positions[j][node_mask[j].squeeze().bool()].tolist()
            pdb_file = os.path.join(args.output_dir, f'molecule_{i * args.batch_size + j}.pdb')
            save_oxygen_pdb(xyz_coordinates, pdb_file)

if __name__ == "__main__":
    main()