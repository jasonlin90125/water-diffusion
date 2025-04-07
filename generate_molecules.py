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
    parser.add_argument('--molecules_per_cluster', type=int, default=10, help='Number of molecules per cluster')
    
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

def save_water_molecule_pdb(positions, atom_decoder, output_dir, molecule_mask=None):
    batch_size = positions.size(0)
    for i in range(batch_size):
        pos = positions[i]
        if molecule_mask is not None:
            pos = pos[molecule_mask[i].bool()]
        
        filename = os.path.join(output_dir, f'cluster_{i}.pdb')
        with open(filename, 'w') as f:
            f.write("REMARK   Generated water cluster\n")
            atom_idx = 1
            mol_idx = 1
            for j in range(0, len(pos), 3):  # Process 3 atoms at a time (1 water molecule)
                # Write oxygen atom
                x, y, z = pos[j] * 10  # Convert to Angstroms
                f.write(f"ATOM  {atom_idx:5d}  O   HOH {mol_idx:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n")
                atom_idx += 1
                
                # Write hydrogen atoms
                for k in range(1, 3):
                    x, y, z = pos[j + k] * 10
                    f.write(f"ATOM  {atom_idx:5d}  H   HOH {mol_idx:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           H\n")
                    atom_idx += 1
                mol_idx += 1
            f.write("END\n")

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

    n_molecules = args.molecules_per_cluster
    n_atoms_total = n_molecules * dataset_info['atoms_per_molecule']
    num_atom_types = dataset_info['num_atom_types']
    atoms_per_mol = dataset_info['atoms_per_molecule'] # Should be 3

    molecule_mask = torch.ones(args.batch_size, n_molecules, device=device)
    atom_mask = molecule_mask.unsqueeze(-1).repeat(1, 1, atoms_per_mol).view(args.batch_size, n_atoms_total).to(device)
    atom_mask_unsqueeze = atom_mask.unsqueeze(-1)
    atom_edge_mask = atom_mask_unsqueeze * atom_mask_unsqueeze.transpose(1, 2)
    diag_mask = ~torch.eye(n_atoms_total, dtype=torch.bool, device=device).unsqueeze(0)
    atom_edge_mask = atom_edge_mask * diag_mask
    atom_edge_mask = atom_edge_mask.to(device)

    fixed_atom_types_flat = torch.tensor([0, 1, 1] * n_molecules, device=device).long() # [N*3]
    one_hot_template = F.one_hot(fixed_atom_types_flat, num_classes=num_atom_types).float() # [N*3, num_types]
    one_hot = one_hot_template.unsqueeze(0).repeat(args.batch_size, 1, 1) # [B, N*3, num_types]
    one_hot = one_hot * atom_mask.unsqueeze(-1) # Apply mask
    
    print(f"Generating {args.n_samples} clusters with {n_molecules} molecules each...")
    os.makedirs(args.output_dir, exist_ok=True)

    generated_count = 0
    with torch.no_grad():
        for i in tqdm(range(0, args.n_samples, args.batch_size)):
            current_batch_size = min(args.batch_size, args.n_samples - generated_count)
            if current_batch_size == 0: break

            # Adjust masks and features for the current batch size
            mol_mask_batch = molecule_mask[:current_batch_size]
            atom_mask_batch = atom_mask[:current_batch_size]
            atom_edge_mask_batch = atom_edge_mask[:current_batch_size]
            one_hot_batch = one_hot[:current_batch_size] # Select batch slice for one_hot
            context_batch = None # No context

            # Sample positions [B, N, 3, 3]
            positions, h_final = model.sample(
                n_samples=current_batch_size,
                n_nodes_mol=n_molecules,
                molecule_mask=mol_mask_batch,
                atom_mask=atom_mask_batch,
                atom_edge_mask=atom_edge_mask_batch,
                one_hot=one_hot_batch,
                context=context_batch
            )

            # Save the generated structures
            save_water_molecule_pdb(
                positions.cpu(),
                dataset_info['atom_decoder'],
                args.output_dir,
                molecule_mask=mol_mask_batch.cpu(),
             )
            print(f"Saved batch {i//args.batch_size + 1}. PDB files starting index {generated_count}.") # Corrected print statement
            generated_count += current_batch_size

if __name__ == "__main__":
    main()