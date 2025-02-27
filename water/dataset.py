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
        self.radius = radius # radius in nM
        self.box_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)  # Box size in nm TODO: Calculate from dataset
        self.masses = torch.tensor([15.999, 1.008, 1.008], dtype=torch.float32)  # O, H, H
        
        with h5py.File(h5_path, 'r') as f:
            pos_dataset = f['positions']
            self.num_frames, atoms_per_frame, _ = pos_dataset.shape
            if atoms_per_frame % 3 != 0:
                raise ValueError("Number of atoms per frame must be divisible by 3")
            
            self.molecules_per_frame = molecules_per_frame = atoms_per_frame // 3
            self.total_samples = self.num_frames * molecules_per_frame
            
        # Precompute mass-weighted centers
        self._precompute_mass_centers()
        
    def _precompute_mass_centers(self):
        """
        Precompute mass-weighted centers for all molecules using memory mapping
        Let f = number of frames, m = number of molecules
        """
        self.com_shape = (self.num_frames, self.molecules_per_frame, 3) # (f, m, 3)
        self.com = np.memmap('com.dat', dtype='float32', mode='w+', shape=self.com_shape)
        
        with h5py.File(self.h5_path, 'r') as f:
            for frame_idx in range(self.num_frames):
                frame_data = f['positions'][frame_idx]  # (m*3, 3)
                reshaped = frame_data.reshape(-1, 3, 3)  # (m, 3, 3)
                
                # Mass-weighted center calculation
                masses_expanded = self.masses.numpy()[:, None]  # (3,) -> (3, 1)
                weighted = reshaped * masses_expanded  # (m, 3, 3)
                summed = weighted.sum(axis=1)  # (m, 3)
                computed = (summed / self.masses.numpy().sum()).astype(np.float32)  # (m, 3)
                self.com[frame_idx] = np.asarray(computed, dtype=np.float32, copy=False)  # (m, 3)
                
        self.com.flush()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        frame_idx, center_idx = divmod(idx, self.molecules_per_frame)
    
        # Load precomputed centers for this frame
        frame_centers = self.com[frame_idx]
        center_pos = frame_centers[center_idx]
        
        # Calculate distances and find neighbors
        distances = np.linalg.norm(frame_centers - center_pos, axis=1)
        neighbor_indices = np.where(distances < self.radius)[0]
        neighbor_centers = frame_centers[neighbor_indices]
        
        # Combine neighbors + center into single array
        all_points = np.vstack([neighbor_centers, center_pos])  # Shape: (N+1,3)
        
        # Compute mean and shift coordinates
        mean_xyz = np.mean(all_points, axis=0)
        shifted_points = all_points - mean_xyz
        
        # Convert to PyTorch tensors
        return torch.from_numpy(shifted_points).float()
        ''' Original code
        frame_idx, center_idx = divmod(idx, self.molecules_per_frame)
        
        # Load precomputed centers for this frame
        frame_centers = self.com[frame_idx]
        center_pos = frame_centers[center_idx]
        
        # Calculate distances using precomputed centers
        distances = np.linalg.norm(frame_centers - center_pos, axis=1)
        neighbor_indices = np.where(distances < self.radius)[0]
        
        # Get neighbor centers (already mass-weighted)
        neighbor_centers = frame_centers[neighbor_indices]
        
        #return {
        #    "node_features": torch.tensor(neighbor_centers, dtype=torch.float32)
        #}
        #return Data(x=torch.tensor(neighbor_centers, dtype=torch.float32))
        return torch.from_numpy(neighbor_centers).float()
        '''

        ''' Boundary conditions
        frame_idx, center_idx = divmod(idx, self.molecules_per_frame)
        frame_centers = self.com[frame_idx]
        box_size = self.box_size
        
        # 1. Wrap coordinates into [0, box_size) for each dimension
        wrapped_centers = np.mod(frame_centers, box_size)
        
        # 2. Verify coordinate bounds before building tree
        if np.any(wrapped_centers < 0) or np.any(wrapped_centers >= box_size):
            raise ValueError("Coordinates still outside box after wrapping")

        # 3. Create PBC-aware KDTree
        tree = cKDTree(wrapped_centers, boxsize=box_size)
        
        # 4. Query neighbors with PBC
        neighbors = tree.query_ball_point(wrapped_centers[center_idx], self.radius)
        neighbors = [i for i in neighbors if i != center_idx]  # Exclude self
        
        neighbor_centers = wrapped_centers[neighbors]
        return torch.from_numpy(neighbor_centers).float()
        '''

    def __del__(self):
        """Clean up memory map resources"""
        if hasattr(self, 'com'):
            del self.com
            try:
                os.remove('com.dat')
            except:
                pass
    
    def test(self):
        # Choose the first frame (frame_idx = 0) and the nth water molecule
        frame_idx = 0
        molecule_idx = 647
        idx = frame_idx * self.molecules_per_frame + molecule_idx

        # Get the sample for this index
        sample = self.__getitem__(idx) # sample.x has shape (num_neighbors, 3)
        print(f"Sample shape: {sample.shape}")

        # Print information about the chosen molecule
        with h5py.File(self.h5_path, 'r') as f:
            frame_data = f['positions'][frame_idx]
            molecule_data = frame_data[molecule_idx*3:(molecule_idx+1)*3]
            
        print(f"Water molecule {molecule_idx} in frame {frame_idx}:")
        print(f"Oxygen  (O): {molecule_data[0]}")
        print(f"Hydrogen (H): {molecule_data[1]}")
        print(f"Hydrogen (H): {molecule_data[2]}")
        
        center_of_mass = self.com[frame_idx, molecule_idx]
        print(f"\nCenter of mass: {center_of_mass}")

        # Print number of neighbors
        num_neighbors = sample.shape[0]
        print(f"\nNumber of neighbors within {self.radius} nm radius: {num_neighbors}")

        # Optionally, print the first few neighbor coordinates
        if num_neighbors > 0:
            print("\nFirst few neighbor coordinates (center of mass):")
            for i in range(num_neighbors):
                print(f"Neighbor {i}: {sample[i]}")

def collate_fn(batch):
# Find max number of nodes in this batch
    max_nodes = max([item.shape[0] for item in batch])
    feat_dim = batch[0].shape[-1]  # Should be 3 for [N, 3]

    # Pad each sample to max_nodes with zeros
    padded_batch = []
    for item in batch:
        padding = max_nodes - item.shape[0]
        padded = torch.cat([
            item, 
            torch.zeros((padding, feat_dim), dtype=item.dtype)
        ], dim=0)
        padded_batch.append(padded)

    # Stack padded tensors [B, max_nodes, 3]
    batch_tensor = torch.stack(padded_batch, dim=0)

    # Create atom mask (1 for real nodes, 0 for padding)
    atom_mask = (batch_tensor.abs().sum(dim=-1) > 0).float()  # [B, max_nodes]

    # Create edge mask (excluding padding and self-edges)
    B, N, _ = batch_tensor.shape
    device = batch_tensor.device
    
    # Valid connections between real atoms
    edge_mask = (atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2))  # [B, N, N]
    
    # Remove self-edges
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    edge_mask *= diag_mask
    
    # Reshape for compatibility
    edge_mask = edge_mask.view(B * N * N, 1)

    return {
        'positions': batch_tensor,
        'atom_mask': atom_mask,
        'edge_mask': edge_mask,
        #'charges': torch.zeros_like(batch_tensor[..., :1]),
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
    
    # Create loaders
    loader_args = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'collate_fn': collate_fn
    }
    
    dataloaders = {
        'train': DataLoader(train, shuffle=True, **loader_args),
        'val': DataLoader(val, **loader_args),
        'test': DataLoader(test, **loader_args)
    }
    
    return dataloaders

if __name__ == '__main__':
    # Example Usage
    torch.manual_seed(0)
    dataset = RadiusGraphDataset("../data/water/water.h5", radius=0.5) # radius in nM
    dataset.test()