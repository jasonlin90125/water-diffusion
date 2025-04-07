import os
import torch
from torch.utils.data import Dataset, Subset, DataLoader  # Standard PyTorch components
#from torch_geometric.data import Data  # PyG data structure
#from torch_geometric.loader import DataLoader  # PyG's optimized loader
#from torch_geometric.nn import MessagePassing
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps
from ase.io import read
import numpy as np
import h5py
from scipy.spatial import cKDTree

class RadiusGraphDataset(Dataset):
    def __init__(self, h5_path, radius):
        self.h5_path = h5_path
        self.radius = radius  # radius in nm
        self.box_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        self.masses = torch.tensor([15.999, 1.008, 1.008], dtype=torch.float32)
        self.atom_types_numeric = torch.tensor([0, 1, 1], dtype=torch.long)
        
        # Calculate inner box boundaries for valid centers
        self.inner_box_min = np.zeros(3, dtype=np.float32) + self.radius
        self.inner_box_max = self.box_size - self.radius
        
        with h5py.File(h5_path, 'r') as f:
            pos_dataset = f['positions']
            #self.num_frames, atoms_per_frame, _ = pos_dataset.shape
            _, atoms_per_frame, _ = pos_dataset.shape
            # Hard code num_frames to limit data size
            self.num_frames = 180
            self.molecules_per_frame = atoms_per_frame // 3
            
        # Precompute mass centers and valid center indices
        self._precompute_mass_centers()
        self._precompute_valid_centers()

    def _precompute_mass_centers(self):
        """Mass center calculation"""
        self.com_shape = (self.num_frames, self.molecules_per_frame, 3)
        self.com = np.memmap('com.dat', dtype='float32', mode='w+', shape=self.com_shape)
        
        with h5py.File(self.h5_path, 'r') as f:
            for frame_idx in range(self.num_frames):
                frame_data = f['positions'][frame_idx]
                reshaped = frame_data.reshape(-1, 3, 3)
                
                masses_expanded = self.masses.numpy()[:, None]
                weighted = reshaped * masses_expanded
                summed = weighted.sum(axis=1)
                computed = (summed / self.masses.numpy().sum()).astype(np.float32)
                self.com[frame_idx] = computed
                
        self.com.flush()

    def _precompute_valid_centers(self):
        """Precompute valid center indices within inner box"""
        self.valid_centers = []
        for frame_idx in range(self.num_frames):
            centers = self.com[frame_idx]
            
            # Find centers within inner box [radius, box_size - radius] in all dimensions
            valid_mask = (
                (centers[:, 0] >= self.inner_box_min[0]) & 
                (centers[:, 0] <= self.inner_box_max[0]) &
                (centers[:, 1] >= self.inner_box_min[1]) & 
                (centers[:, 1] <= self.inner_box_max[1]) &
                (centers[:, 2] >= self.inner_box_min[2]) & 
                (centers[:, 2] <= self.inner_box_max[2])
            )
            
            self.valid_centers.extend(
                [(frame_idx, idx) for idx in np.where(valid_mask)[0]]
            )
            
        self.total_samples = len(self.valid_centers)

    def __len__(self):
        return self.total_samples

    '''
    def __getitem__(self, idx):
        frame_idx, center_idx = self.valid_centers[idx]
        frame_centers = self.com[frame_idx]
        center_pos = frame_centers[center_idx]
        
        # Regular distance calculation (no periodic boundaries)
        distances = np.linalg.norm(frame_centers - center_pos, axis=1)
        neighbor_indices = np.where(distances < self.radius)[0]
        neighbor_centers = frame_centers[neighbor_indices]

        # Combine neighbors + center
        all_points = np.vstack([neighbor_centers, center_pos])
        mean_xyz = np.mean(all_points, axis=0)
        shifted_points = all_points - mean_xyz
        
        return {
            'coordinates': torch.from_numpy(shifted_points).float(),
            'num_nodes': len(shifted_points)
        }
    '''

    def __getitem__(self, idx):
        frame_idx, center_idx = self.valid_centers[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            frame_data = f['positions'][frame_idx]  # (num_atoms, 3)
        
        reshaped = frame_data.reshape(-1, 3, 3)  # (num_molecules, 3 atoms, 3 coordinates)
        
        # Get center molecule's atoms
        center_molecule = reshaped[center_idx]  # (3, 3)
        
        # Find neighboring molecules using COM distances
        coms = np.mean(reshaped, axis=1)  # (num_molecules, 3)
        distances = np.linalg.norm(coms - coms[center_idx], axis=1)
        neighbor_indices = np.where(distances < self.radius)[0]
        
        # Collect all atoms from neighbors + center
        neighbor_molecules = reshaped[neighbor_indices]  # (num_neighbors, 3, 3)
        all_molecules = np.vstack([neighbor_molecules, center_molecule[np.newaxis]])  # (num_neighbors+1, 3, 3)
        
        # Flatten to atomic coordinates and center
        all_atoms = all_molecules.reshape(-1, 3)  # (3*(num_neighbors+1), 3)
        mean_xyz = np.mean(all_atoms, axis=0)
        shifted_atoms = all_atoms - mean_xyz

        shifted_molecules = shifted_atoms.reshape(-1, 3, 3)  # ((num_neighbors+1), 3, 3)

        # Atom types
        atom_types = self.atom_types_numeric.repeat(len(shifted_molecules)) # [O, H, H, O, H, H, ...]

        return {
            'coordinates': torch.from_numpy(shifted_molecules).float(), # shape: (num_molecules, 3, 3)
            'num_nodes': len(shifted_molecules), # Number of molecules in this sample
            'atom_types_flat': atom_types # shape: (num_molecules * 3)
        }

    def __del__(self):
        """Clean up memory map resources"""
        if hasattr(self, 'com'):
            del self.com
            try:
                os.remove('com.dat')
            except:
                pass
    
    def test(self):
        # Example test to visualize a sample
        idx = 0

        # Get the sample for this index
        sample = self.__getitem__(idx)
        print(f"Coordinates: {sample['coordinates']}")
        # Save the sample coordinates to PDB
        save_pdb_from_tensor(sample['coordinates'], 'sample.pdb')
        print(f"Number of nodes: {sample['num_nodes']}")
        print(f"Total samples: {self.total_samples}")

def save_pdb_from_tensor(coords, filename):
    with open(filename, 'w') as f:
        atom_index = 1
        for mol_idx, molecule in enumerate(coords):
            # O atom
            x, y, z = molecule[0]
            f.write(f"ATOM  {atom_index:5d}  O   HOH {mol_idx+1:4d}    {x*10:8.3f}{y*10:8.3f}{z*10:8.3f}  1.00  0.00           O\n")
            atom_index += 1
            # H atoms
            for h_idx in range(1, 3):
                x, y, z = molecule[h_idx]
                f.write(f"ATOM  {atom_index:5d}  H   HOH {mol_idx+1:4d}    {x*10:8.3f}{y*10:8.3f}{z*10:8.3f}  1.00  0.00           H\n")
                atom_index += 1

def collate_fn(batch):
    num_molecules_list = [item['num_nodes'] for item in batch]
    max_molecules = max(num_molecules_list)
    atoms_per_mol = batch[0]['coordinates'].shape[1] # Should be 3
    coords_dim = batch[0]['coordinates'].shape[2]   # Should be 3
    num_atom_types = 2 # O, H

    batch_positions = []
    batch_atom_types_flat = []
    batch_molecule_mask = []

    for i, item in enumerate(batch):
        mol_pos = item['coordinates'] # [N_mol, 3, 3]
        num_mols = item['num_nodes']
        padding_mols = max_molecules - num_mols

        # Pad positions tensor
        padded_pos = torch.cat([
            mol_pos,
            torch.zeros((padding_mols, atoms_per_mol, coords_dim), dtype=mol_pos.dtype)
        ], dim=0) # [max_N, 3, 3]
        batch_positions.append(padded_pos)

        # Molecule mask
        mask = torch.zeros(max_molecules, dtype=torch.float32)
        mask[:num_mols] = 1.0
        batch_molecule_mask.append(mask)

        # Pad flat atom types (careful with indices)
        atom_types = item['atom_types_flat'] # [N_mol * 3]
        padded_atoms = torch.cat([
            atom_types,
            torch.zeros(padding_mols * atoms_per_mol, dtype=atom_types.dtype)
        ], dim=0) # [max_N * 3]
        batch_atom_types_flat.append(padded_atoms)

    # Stack tensors
    positions = torch.stack(batch_positions, dim=0) # [B, max_N, 3, 3]
    molecule_mask = torch.stack(batch_molecule_mask, dim=0) # [B, max_N]

    # --- Create Atom-Level Masks and Features ---
    B, max_N = molecule_mask.shape
    max_atoms_total = max_N * atoms_per_mol

    # Atom mask [B, max_N * 3]
    atom_mask = molecule_mask.unsqueeze(-1).repeat(1, 1, atoms_per_mol).view(B, max_atoms_total)

    # One-hot encoding for atoms [B, max_N * 3, num_atom_types]
    atom_types_flat_tensor = torch.stack(batch_atom_types_flat, dim=0).long() # [B, max_N * 3]
    one_hot = torch.nn.functional.one_hot(atom_types_flat_tensor, num_classes=num_atom_types).float()
    one_hot = one_hot * atom_mask.unsqueeze(-1) # Apply atom mask

    # --- Create Atom-Level Edge Mask ---
    # [B, max_atoms_total, max_atoms_total]
    atom_mask_unsqueeze = atom_mask.unsqueeze(-1) # [B, max_atoms_total, 1]
    edge_mask = atom_mask_unsqueeze * atom_mask_unsqueeze.transpose(1, 2) # [B, max_atoms_total, max_atoms_total]

    # Remove self-loops
    diag_mask = ~torch.eye(max_atoms_total, dtype=torch.bool, device=positions.device).unsqueeze(0)
    edge_mask = edge_mask * diag_mask

    return {
        'positions': positions,         # [B, max_N, 3, 3] - Molecule-centric positions
        'molecule_mask': molecule_mask, # [B, max_N]       - Mask for molecules
        'atom_mask': atom_mask,         # [B, max_N * 3]   - Mask for atoms
        'one_hot': one_hot,             # [B, max_N * 3, n_atom_types] - Atom features
        'edge_mask': edge_mask,         # [B, max_N * 3, max_N * 3]   - Atom edge mask
        'num_nodes_per_sample': torch.tensor(num_molecules_list) # Original number of molecules per sample
    }

def retrieve_dataloaders(cfg):
    if 'water' not in cfg.dataset:
        raise ValueError(f'Unsupported dataset: {cfg.dataset}')

    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(f"Dataset file {cfg.dataset_path} not found")
    
    dataset = RadiusGraphDataset(cfg.dataset_path, radius=cfg.radius)
    
    # Split dataset
    split = getattr(cfg, 'split', (0.8, 0.1, 0.1))
    n = len(dataset)
    train_end = int(n * split[0])
    val_end = train_end + int(n * split[1])
    
    train = Subset(dataset, range(0, train_end))
    val = Subset(dataset, range(train_end, val_end))
    test = Subset(dataset, range(val_end, n))

    print('Training dataset has', len(train), 'samples')
    print('Validation dataset has', len(val), 'samples')
    print('Test dataset has', len(test), 'samples')
    
    # Create loaders
    loader_args = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'collate_fn': collate_fn
    }
    
    dataloaders = {
        'train': DataLoader(train, shuffle=True, **loader_args),
        'valid': DataLoader(val, **loader_args),
        'test': DataLoader(test, **loader_args)
    }
    
    return dataloaders

if __name__ == '__main__':
    # Example Usage
    torch.manual_seed(0)
    dataset = RadiusGraphDataset("../data/water/water.h5", radius=0.5) # radius in nM
    dataset.test()