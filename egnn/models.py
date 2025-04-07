import torch
import torch.nn as nn
from egnn.egnn_new import EGNN, GNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np


class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

class EGNN_dynamics_water(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, num_atom_types,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        self.num_atom_types = num_atom_types
        self.atoms_per_molecule = 3 # hardcoded for water

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            '''
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)
            '''
            raise NotImplementedError("GNN dynamics not fully adapted for [B, N, 3, 3] structure.")

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, one_hot_per_batch, # [B, M, Types]
                    molecule_mask_per_batch, # [B, N] - Add this
                    atom_mask_per_batch,     # [B, M]
                    atom_edge_mask_per_batch,# [B, M, M]
                    context_per_batch):
        def fwd(time, state_x): # state_x is [B, N, 3, 3]
            # ... (reshaping x, calculating h_atoms, reshaping masks, getting edges) ...
            # ... (as before) ...
            bs = state_x.shape[0]
            max_n_mol = state_x.shape[1]
            max_n_atoms = max_n_mol * self.atoms_per_molecule

            h_atoms = one_hot_per_batch.to(self.device) # [B, M, Types]

            if np.prod(time.size()) == 1:
                h_time = torch.empty(bs, max_n_atoms, 1, device=self.device).fill_(time.item())
            else:
                h_time = time.view(bs, 1, 1).repeat(1, max_n_atoms, 1)
            h_atoms = torch.cat([h_atoms, h_time], dim=2) # [B, M, F]

            if context_per_batch is not None:
                 h_atoms = torch.cat([h_atoms, context_per_batch.to(self.device)], dim=2)

            x_atoms = state_x.view(bs * max_n_atoms, self.n_dims)
            h_atoms_in = h_atoms.view(bs * max_n_atoms, -1)
            atom_mask_in = atom_mask_per_batch.view(bs * max_n_atoms, 1)
            atom_edge_mask_dense = atom_edge_mask_per_batch # Keep dense [B, M, M]
            edges = self.get_adj_matrix(max_n_atoms, bs, self.device)

            # === Pass molecule_mask_per_batch to _forward ===
            vel_pred_mol = self._forward(
                time,
                x_atoms,
                h_atoms_in,
                edges,
                atom_mask_in, # Atom mask [B*M, 1]
                molecule_mask_per_batch, # Pass molecule mask [B, N]
                atom_edge_mask_dense, # Pass dense edge mask [B, M, M]
                context=None
            )

            return vel_pred_mol
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, x_atoms, h_atoms, edges,
                atom_mask, # [B*M, 1]
                molecule_mask, # [B, N] - Added this
                atom_edge_mask_dense, # [B, M, M]
                context):
        # ... (get bs, max_n_atoms, max_n_mol from shapes) ...
        bs = molecule_mask.shape[0]
        max_n_mol = molecule_mask.shape[1]
        max_n_atoms = max_n_mol * self.atoms_per_molecule

        if self.mode == 'egnn_dynamics':
            h_final_atoms, x_final_atoms = self.egnn(
                h=h_atoms,
                x=x_atoms,
                edge_index=edges,
                node_mask=atom_mask,
                edge_mask=atom_edge_mask_dense
            )
            vel_atoms = (x_final_atoms - x_atoms) * atom_mask
        else:
             raise Exception("Wrong mode %s" % self.mode) # GNN mode check

        # ... (Reshape vel_atoms to vel_mol) ...
        vel_mol = vel_atoms.view(bs, max_n_mol, self.atoms_per_molecule, self.n_dims)

        # ... (Center velocity using atom_mask) ...
        atom_mask_reshaped = atom_mask.view(bs, max_n_atoms, 1) # Use the passed atom_mask
        vel_atoms_masked = vel_atoms.view(bs, max_n_atoms, self.n_dims) * atom_mask_reshaped
        sum_vel = torch.sum(vel_atoms_masked, dim=1, keepdim=True)
        num_atoms_per_sample = torch.sum(atom_mask_reshaped, dim=1, keepdim=True)
        mean_vel = sum_vel / torch.clamp(num_atoms_per_sample, min=1)
        vel_atoms_centered = vel_atoms.view(bs, max_n_atoms, self.n_dims) - mean_vel
        vel_mol_centered = vel_atoms_centered.view(bs, max_n_mol, self.atoms_per_molecule, self.n_dims)


        # === Use the passed molecule_mask for final masking ===
        # Unsqueeze molecule_mask [B, N] -> [B, N, 1, 1] for broadcasting
        molecule_mask_unsqueezed = molecule_mask.unsqueeze(-1).unsqueeze(-1)
        vel_mol_final = vel_mol_centered * molecule_mask_unsqueezed # Apply correct mask

        # ... (NaN check) ...
        if torch.any(torch.isnan(vel_mol_final)):
            print('Warning: detected nan in dynamics output, resetting to zero.')
            vel_mol_final = torch.zeros_like(vel_mol_final)

        return vel_mol_final

    # get_adj_matrix needs to work with the total number of atoms
    def get_adj_matrix(self, n_nodes_total, batch_size, device):
        if n_nodes_total in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes_total]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes_total):
                        for j in range(n_nodes_total):
                            if i != j: # Avoid self-loops in edge list
                                rows.append(i + batch_idx * n_nodes_total)
                                cols.append(j + batch_idx * n_nodes_total)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                # print(f"Generated edges for {n_nodes_total} nodes, batch {batch_size}. Num edges: {len(rows)}")
                return edges
        else:
            self._edges_dict[n_nodes_total] = {}
            return self.get_adj_matrix(n_nodes_total, batch_size, device)