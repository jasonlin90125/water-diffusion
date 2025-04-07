import torch
import numpy as np

def gaussian_KL_for_dimension_batch(q_mu, q_sigma, p_mu, p_sigma, d):
    # q_mu [B, N_atoms, 3], q_sigma [B,], p_mu [B, N_atoms, 3], p_sigma [B,]
    # d = degrees of freedom [B,]
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2) # [B,]
    term1 = d * torch.log(p_sigma / q_sigma)             # [B,]
    term2 = 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) # [B,]
    term3 = -0.5 * d                                     # [B,]
    return term1 + term2 + term3

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x

'''
def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x
'''
def remove_mean_with_mask(x, node_mask):
    # x: [B, N, D] or [B, N, 3, 3]
    # node_mask: [B, N, 1] or [B, N] (needs broadcasting)
    if x.dim() == 4: # Handle [B, N, 3, 3] by calculating COM
        # node_mask should be molecule_mask [B, N]
        if node_mask.dim() == 2: node_mask = node_mask.unsqueeze(-1).unsqueeze(-1) #[B, N, 1, 1]
        num_molecules = node_mask.sum(dim=1, keepdim=True).clamp(min=1) #[B, 1, 1, 1]
        # Calculate COM of each molecule, then mean COM over valid molecules
        com_per_molecule = x.mean(dim=2, keepdim=True) # [B, N, 1, 3]
        mean_com = (com_per_molecule * node_mask).sum(dim=1, keepdim=True) / num_molecules # [B, 1, 1, 3]
        x = x - mean_com # Broadcast and subtract
        x = x * node_mask # Ensure padded values remain zero
    elif x.dim() == 3: # Standard atom-level centering
        # node_mask should be atom_mask [B, N, 1]
        if node_mask.dim() == 2: node_mask = node_mask.unsqueeze(-1) # Ensure [B, N, 1]
        N = node_mask.sum(dim=1, keepdim=True).clamp(min=1) # [B, 1, 1]
        mean = torch.sum(x*node_mask, dim=1, keepdim=True) / N # [B, 1, D]
        x = (x - mean) * node_mask
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions")
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4

'''
def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'
'''
def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
     if x.dim() == 4: # Handle [B, N, 3, 3]
         if node_mask.dim()==2: node_mask = node_mask.unsqueeze(-1).unsqueeze(-1) # Mol mask [B,N,1,1]
         com_per_molecule = x.mean(dim=2, keepdim=True) # [B, N, 1, 3]
         num_molecules = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
         mean_com = (com_per_molecule * node_mask).sum(dim=1) / num_molecules # [B, 3]
         error = mean_com.abs().max().item()
         assert error < 1e-2, f'Mean COM is not zero, error {error}'
     elif x.dim() == 3: # Standard atom centering
         if node_mask.dim() == 2: node_mask = node_mask.unsqueeze(-1)
         N = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
         mean = torch.sum(x*node_mask, dim=1, keepdim=True) / N
         error = mean.abs().max().item()
         assert error < 1e-2, f'Mean is not zero, error {error}'
     else:
          raise ValueError("Input tensor must have 3 or 4 dimensions")


def assert_correctly_masked(variable, node_mask):
    masked_max_abs_value = (variable * (1 - node_mask)).abs().max().item()
    assert masked_max_abs_value < 1e-4, \
        f'Variables not masked properly. Masked max abs value: {masked_max_abs_value}'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked
