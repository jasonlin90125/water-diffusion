import os
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps
from ase.io import read
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
import numpy as np

class RadiusGraphDataset(Dataset):
    def __init__(self, h5_path, radius): 
        self.h5_path = h5_path
        self.radius = radius # radius in nM
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
        
        # Calculate distances using precomputed centers
        distances = np.linalg.norm(frame_centers - center_pos, axis=1)
        neighbor_indices = np.where(distances < self.radius)[0]
        
        # Get neighbor centers (already mass-weighted)
        neighbor_centers = frame_centers[neighbor_indices]
        
        return {
            "node_features": torch.tensor(neighbor_centers, dtype=torch.float32)
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
        # Choose the first frame (frame_idx = 0) and the 1000th water molecule
        frame_idx = 0
        molecule_idx = 1000
        idx = frame_idx * self.molecules_per_frame + molecule_idx

        # Get the sample for this index
        sample = self.__getitem__(idx)

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
        num_neighbors = sample["node_features"].shape[0]
        print(f"\nNumber of neighbors within {self.radius} nm radius: {num_neighbors}")

        # Optionally, print the first few neighbor coordinates
        if num_neighbors > 0:
            print("\nFirst few neighbor coordinates (center of mass):")
            for i in range(min(5, num_neighbors)):
                print(f"Neighbor {i}: {sample['node_features'][i]}")

if __name__ == '__main__':
    # Example Usage
    torch.manual_seed(0)
    dataset = RadiusGraphDataset("../data/water/water.h5", radius=0.5) # radius in nM
    dataset.test()

'''
import torch
import numpy as np
import h5py
import os
import warnings
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class WaterDataset(Dataset):
    def __init__(self, filename, use_com=True, box_size=None, 
                 random_box=True, min_waters=3, mass_o=16.0, mass_h=1.0):
        self.filename = filename
        self.use_com = use_com
        self.box_size = np.array(box_size) if box_size else None
        self.random_box = random_box
        self.min_waters = min_waters
        self.mass_o = mass_o
        self.mass_h = mass_h
        
        with h5py.File(filename, 'r') as f:
            self.n_frames = f['positions'].shape[0]
            self.n_waters = f['positions'].shape[1] // 3
            
        self._validate_parameters()

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        with h5py.File(self.filename, 'r') as f:
            frame_data = f['positions'][idx]
            
        waters = frame_data.reshape(-1, 3, 3)
        
        # Calculate positions (COM or oxygen)
        if self.use_com:
            o_xyz = waters[:, 0]
            h1_xyz = waters[:, 1]
            h2_xyz = waters[:, 2]
            com = (self.mass_o * o_xyz + self.mass_h * (h1_xyz + h2_xyz)) / (self.mass_o + 2*self.mass_h)
            positions = com
        else:
            positions = waters[:, 0]

        # Calculate box dimensions from coordinates
        all_coords = waters.reshape(-1, 3)
        frame_box_dim = np.ptp(all_coords, axis=0)  # max - min per dimension
        
        # Validate box dimensions
        self._validate_box(frame_box_dim, idx)

        # Apply spatial subsetting
        if self.box_size is not None:
            positions = self._apply_spatial_subset(positions, frame_box_dim)
        elif self.min_waters:
            positions = self._random_subset(positions)
            
        return torch.tensor(positions, dtype=torch.float32)

    def _validate_parameters(self):
        if self.box_size is not None and self.min_waters is not None:
            raise ValueError("Use either box_size or min_waters, not both")
            
    def _validate_box(self, box_dim, idx):
        min_size = 1.4  # 14Å in nm
        if np.any(box_dim < min_size):
            warnings.warn(f"Small box {box_dim} at frame {idx} (<14Å)")
        if np.any(box_dim > 100.0):  # 100nm = 1μm
            warnings.warn(f"Large box {box_dim} at frame {idx} (>1μm)")

    def _random_subset(self, positions):
        n_waters = positions.shape[0]
        select = np.random.choice(n_waters, 
                                max(self.min_waters, np.random.randint(3, n_waters)), 
                                replace=False)
        return positions[select]

    def _apply_spatial_subset(self, positions, frame_box_dim):
        if self.box_size is None:
            return positions
            
        if np.any(self.box_size > frame_box_dim):
            raise ValueError(f"Requested box {self.box_size} exceeds frame dimensions {frame_box_dim}")

        if self.random_box:
            max_pos = frame_box_dim - self.box_size
            origin = np.random.uniform(0, max_pos.clip(min=0), 3)
        else:
            origin = (frame_box_dim - self.box_size) / 2

        mask = np.all((positions >= origin) & 
                     (positions <= origin + self.box_size), axis=1)
        return positions[mask]

class PreprocessWater:
    def collate_fn(self, batch):
        max_atoms = max([x.shape[0] for x in batch])
        padded_batch = []
        masks = []
        for x in batch:
            n_atoms = x.shape[0]
            padding = torch.zeros((max_atoms - n_atoms, 3), dtype=x.dtype)
            padded = torch.cat([x, padding], dim=0)
            padded_batch.append(padded)
            mask = torch.cat([torch.ones(n_atoms), torch.zeros(max_atoms - n_atoms)])
            masks.append(mask)
        return torch.stack(padded_batch), torch.stack(masks)

def retrieve_dataloaders(cfg):
    if 'water' not in cfg.dataset:
        raise ValueError(f'Unsupported dataset: {cfg.dataset}')

    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(f"Dataset file {cfg.dataset_path} not found")
    
    dataset = WaterDataset(
        cfg.dataset_path,
        use_com=getattr(cfg, 'use_com', True),
        box_size=getattr(cfg, 'box_size', None),
        random_box=getattr(cfg, 'random_box', True),
        min_waters=getattr(cfg, 'min_waters', None)
    )
    
    # Split dataset
    split = getattr(cfg, 'split', (0.7, 0.2, 0.1))
    if getattr(cfg, 'chronological', True):
        n = len(dataset)
        train = Subset(dataset, range(0, int(n*split[0])))
        val = Subset(dataset, range(int(n*split[0]), int(n*(split[0]+split[1]))))
        test = Subset(dataset, range(int(n*(split[0]+split[1])), n))
    else:
        train, val, test = random_split(dataset, [
            int(len(dataset)*split[0]),
            int(len(dataset)*split[1]),
            len(dataset) - int(len(dataset)*(split[0]+split[1]))
        ])
    
    # Create loaders
    preprocess = PreprocessWater()
    loader_args = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'collate_fn': preprocess.collate_fn
    }
    
    dataloaders = {
        'train': DataLoader(train, shuffle=not getattr(cfg, 'chronological', True), **loader_args),
        'val': DataLoader(val, **loader_args),
        'test': DataLoader(test, **loader_args)
    }
    
    return dataloaders, None

def load_dataset(filename="../data/water/water.h5"):
    """ Load the water MD dataset """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found; download it from https://storage.googleapis.com/boltzmann_inpainting/water.h5")
    return h5py.File(filename, "r")

def retrieve_dataloaders(cfg):
    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets
'''