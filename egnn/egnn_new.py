from torch import nn
import torch
import math

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            #nn.Linear(input_edge + edges_in_d, hidden_nf),
            nn.Linear(input_edge + 3*edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            #print('Source shape', source.shape) # [B*N*N, 256]
            #print('Target shape', target.shape) # [B*N*N, 256]
            #print('Edge attr shape', edge_attr.shape) # [B*N*N, 6]
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        #print('edge_attr.shape', edge_attr.shape) # [B*N*N, 256]
        #print('row.shape', row.shape) # [B*N*N]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        #print('h shape init', h.shape) # [B*N, 256]
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        #print('h shape after node_model', h.shape) # [B*N, 256]
        #print('node_mask shape', node_mask.shape) # [B*N, 1, 1]
        if node_mask is not None:
            #h = h * node_mask
            h = h * node_mask.squeeze(-1) # [B*N, 256] * [B*N, 1]
        #print('h shape after node_mask:', h.shape) # [B*N, 256]
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + 3*edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        # coord is x
        row, col = edge_index
        #print('row.shape', row.shape) # [B*N*N]
        #print('col.shape', col.shape) # [B*N*N]
        #print('coord.shape', coord.shape) # [B*N, 3, 3]
        #print('h.shape', h.shape) # [B*N, B*N, 256]
        #print('h[row].shape', h[row].shape) # [B*N*N, 256]
        #print('h[col].shape', h[col].shape) # [B*N*N, 256]
        #print('edge_attr.shape', edge_attr.shape) # [B*N*N, 6]
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1) # [B*N*N, 256*2 + 6]
        if self.tanh:
            #print('coord_diff.shape', coord_diff.shape) # [B*N*N, 3, 3]
            #print('torch.tanh(self.coord_mlp(input_tensor)).shape', torch.tanh(self.coord_mlp(input_tensor)).shape) # [B*N*N, 1]
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)).unsqueeze(1) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            #print('edge_mask.shape', edge_mask.shape) # [B*N*N, 1]
            #print('trans.shape', trans.shape) # [B*N*N, 3, 3]
            trans = trans * edge_mask.unsqueeze(1)
            #print('trans.shape', trans.shape) # [B*N*N, 3, 3]
        
        #print('trans.shape', trans.shape) # [B*N*N, 3, 3]
        #print('row.shape', row.shape) # [B*N*N]
        #print('coord.shape', coord.shape) # [B*N, 3, 3]
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method) # need to unsqueeze(1)
        #print('agg.shape', agg.shape) # [B*N, 3, 3]
        #print('coord.shape', coord.shape) # [B*N, 3, 3]
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)

        #print('coord.shape', coord.shape) # [B*N, 3, 3]
        #print('node_mask.shape', node_mask.shape) # [B*N, 1, 1]
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        #print('distances.shape', distances.shape) # [B*N*N, 3]
        #print('edge_attr.shape', edge_attr.shape) # [B*N*N, 3]
        #print('coord_diff.shape', coord_diff.shape) # [B*N*N, 3, 3]
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        #print('h.shape after GCL', h.shape) # [B*N, 256]
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask) # [B*N, 3, 3]
        #print('x.shape after equiv', x.shape) # [B*N, 3, 3]

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask.squeeze(1)
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        # print('x.shape', x.shape) # [B*N, 3, 3]
        # print('h.shape', h.shape) # [B*N, 3]
        # print('node_mask.shape', node_mask.shape) # [B*N, 1, 1]
        # print('edge_mask.shape', edge_mask.shape) # [B*N*N, 1]

        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h) # [B*N, 256]
        #print('distances.shape', distances.shape) # [B*N*N, 3]
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask.squeeze(1)
        return h, x

# Not used currently
class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col] 
    #radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    radial = torch.sum((coord_diff) ** 2, 1)
    norm = torch.sqrt(radial + 1e-8)
    # print(coord_diff.shape) # [B*N*N, 3, 3]
    # print(norm.unsqueeze(1).shape) # [B*N*N, 1, 3]
    coord_diff = coord_diff/(norm.unsqueeze(1) + norm_constant)
    return radial, coord_diff

'''
def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    # print(result_shape) # [B*N, 256] or [B*N, 3]
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    print('segment_ids.shape', segment_ids.shape) # [B*N*N]
    print('data.shape', data.shape) # [B*N*N, 3]
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    print(segment_ids.shape) # [B*N*N, 3]
    print(data.shape) # [B*N*N, 3, 3]
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

'''

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    if not segment_ids.dim() == 1:
        # Ensure segment_ids is always 1D as expected (the row indices)
        raise ValueError(f"segment_ids is expected to be 1D, but got shape {segment_ids.shape}")
    if not data.shape[0] == segment_ids.shape[0]:
         raise ValueError(f"Dimension 0 of data ({data.shape[0]}) must match "
                          f"length of segment_ids ({segment_ids.shape[0]})")

    # 1. Determine the shape for the result tensor
    #    Result has `num_segments` rows and keeps all other dimensions of `data`.
    result_shape = (num_segments,) + data.shape[1:] # e.g., (N, D1) or (N, D1, D2)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.

    # 2. Dynamically prepare segment_ids for scatter_add_
    #    We need to reshape segment_ids [E] -> [E, 1, ..., 1] to have the same
    #    number of dimensions as `data`. The number of singleton dimensions added
    #    is `data.dim() - 1`.
    view_shape = (segment_ids.shape[0],) + (1,) * (data.dim() - 1)
    segment_ids_reshaped = segment_ids.view(view_shape)

    # 3. Expand the reshaped segment_ids to match the full shape of `data`.
    #    This creates the index tensor needed by scatter_add_ for N-D data.
    #    Example: if data is [E, D1, D2], segment_ids_reshaped is [E, 1, 1].
    #    expand_as(data) makes it [E, D1, D2], where each slice [:, d1, d2]
    #    contains the original segment IDs.
    segment_ids_expanded = segment_ids_reshaped.expand_as(data)

    # 4. Perform the scatter add operation using the correctly expanded indices
    result.scatter_add_(0, segment_ids_expanded, data)

    # 5. Apply normalization (logic remains the same, but uses expanded ids for counting)
    if aggregation_method == 'sum':
        result = result / (normalization_factor + 1e-8) # Add epsilon for stability

    elif aggregation_method == 'mean':
        norm = data.new_zeros(result_shape)
        ones_to_sum = torch.ones_like(data)
        # Use the same expanded segment_ids shape for counting
        norm.scatter_add_(0, segment_ids_expanded, ones_to_sum)
        norm = torch.clamp(norm, min=1) # Avoid division by zero
        result = result / norm

    return result