from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: models.EGNN_dynamics_QM9, 
            in_node_nf: int, 
            n_dims: int,
            timesteps: int = 1000, 
            parametrization='eps', 
            noise_schedule='learned',
            noise_precision=1e-4, 
            loss_type='vlb', 
            norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), 
            include_charges=True):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        self.atoms_per_molecule = dynamics.atoms_per_molecule # Get from dynamics model

        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        assert isinstance(dynamics, models.EGNN_dynamics_water), "Dynamics model must be EGNN_dynamics_water for this setup"

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    '''
    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out
    '''
    def phi(self, x, t, one_hot, molecule_mask, atom_mask, atom_edge_mask, context): # atom_edge_mask is dense [B,M,M]
        fwd_dynamics = self.dynamics.wrap_forward(
            one_hot, # Pass one_hot
            molecule_mask,
            atom_mask, # Pass atom mask [B, M]
            atom_edge_mask, # Pass dense atom edge mask [B, M, M]
            context
        )
        net_out = fwd_dynamics(t, x) # Returns predicted velocity [B, N, 3, 3]
        return net_out

    '''
    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)
    '''
    def inflate_batch_array(self, array, target):
        # Needs to handle target shape [B, N, 3, 3]
        # Inflate array [B,] to [B, 1, 1, 1]
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    '''
    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims
    '''
    def subspace_dimensionality(self, atom_mask):
        # Dimensionality based on total number of *atoms* minus COM constraint.
        # atom_mask shape [B, max_n_atoms]
        number_of_atoms = torch.sum(atom_mask, dim=1) # [B,]
        return (number_of_atoms - 1) * self.n_dims

    '''
    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px
    '''
    def normalize(self, x, atom_mask):
        # x shape [B, N, 3, 3]
        # Normalize coordinates only
        x_norm = x / self.norm_values[0]
        # Reshape atom_mask [B, max_n_atoms] for subspace calculation
        delta_log_px = -self.subspace_dimensionality(atom_mask) * np.log(self.norm_values[0])
        return x_norm, delta_log_px

    '''
    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int
    '''
    def unnormalize(self, x):
        # x shape [B, N, 3, 3]
        x_unnorm = x * self.norm_values[0]
        return x_unnorm

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    '''
    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x + kl_distance_h
    '''
    def kl_prior(self, x, atom_mask):
        # Calculate KL divergence based on the normalized coordinates x [B, N, 3, 3]
        # Treat it as atom coordinates for KL calculation purposes
        bs, n_mol, _, _ = x.shape
        n_atoms_total = n_mol * self.atoms_per_molecule
        x_atoms = x.view(bs, n_atoms_total, self.n_dims) # Reshape for KL

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((x.size(0), 1), device=x.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, x_atoms) # Use x_atoms shape

        # Compute means.
        mu_T = alpha_T * x_atoms

        # Compute standard deviations (needs careful handling for subspace)
        sigma_T_val = self.sigma(gamma_T, mu_T).squeeze() # [B,]

        # Compute KL for x-part (atom coordinates)
        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(sigma_T_val) # Sigma is scalar per batch item
        subspace_d = self.subspace_dimensionality(atom_mask) # [B,]

        # gaussian_KL_for_dimension needs adjustment if sigma varies per batch item
        # Simplified: Assume sigma_T is scalar for broadcasting in KL calculation.
        # This might be an approximation if gamma varies significantly per batch item (unlikely).
        kl_distance_x = utils.gaussian_KL_for_dimension_batch(mu_T, sigma_T_val, zeros, ones, d=subspace_d)

        # No feature part KL
        return kl_distance_x
    
    def predict_eps_from_vel(self, vel, zt, alpha_t, sigma_t):
        """Converts predicted velocity v_pred to predicted noise eps_pred."""
        # Based on x_pred = zt/alpha_t - sigma_t/alpha_t * eps_pred
        # and v_pred = (x_pred - alpha_t*x) / sigma_t  (approximation from DDPM)
        # OR more directly from the score: score = -eps/sigma
        # And v_pred is related to the score. Let's use the DDIM-like derivation:
        # x_pred = alpha_t * x + sigma_t * v_pred (if v_pred parameterization) NO
        # From score matching: v = -sigma * score = sigma * eps / sigma = eps
        # If dynamics predicts velocity `v`, then `v = (dx/dt) / sigma_dot`? Complex.

        # Let's assume the dynamics model directly predicts `v = eps` for simplicity,
        # aligning with score = -eps/sigma and dynamics output proportional to score.
        # This is a common simplification/parameterization choice.
        # If the dynamics truly predicted velocity dx/dt, the conversion is more complex.
        # Assuming dynamics output `net_out` is implicitly `eps_pred`:
        # return vel # If dynamics directly predicts eps

        # If dynamics predicts velocity v_t, and z_t = alpha_t*x + sigma_t*eps
        # d(z_t)/dt = alpha_dot*x + sigma_dot*eps
        # v_t = dynamics(z_t, t) approx d(z_t)/dt?
        # Need to relate v_t back to eps. Rearranging z_t: x = (z_t - sigma_t*eps) / alpha_t
        # v_t = alpha_dot/alpha_t * (z_t - sigma_t*eps) + sigma_dot*eps
        # v_t = (alpha_dot/alpha_t)*z_t + (sigma_dot - alpha_dot/alpha_t*sigma_t)*eps
        # eps_pred = (v_t - (alpha_dot/alpha_t)*z_t) / (sigma_dot - alpha_dot/alpha_t*sigma_t)
        # This requires alpha_dot, sigma_dot. Let's stick to the simpler parameterization
        # where the network output `vel` is treated as `eps_pred`.

        return vel # Assuming dynamics output is parameterized as epsilon

    '''
    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred
    '''
    def compute_x_pred(self, net_out_eps, zt, gamma_t):
        # This function now takes predicted epsilon
        sigma_t = self.sigma(gamma_t, target_tensor=net_out_eps)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out_eps)
        eps_t = net_out_eps
        x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        return x_pred
    
    '''
    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error
    '''
    def compute_error(self, net_out_eps, gamma_t, eps):
        eps_t = net_out_eps
        # Loss per coordinate dimension
        error = sum_except_batch((eps - eps_t) ** 2) / self.n_dims
        return error

    '''
    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))
    '''
    def log_constants_p_x_given_z0(self, x, atom_mask):
        batch_size = x.size(0)
        degrees_of_freedom_x = self.subspace_dimensionality(atom_mask) #[B,]

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # log_sigma_x = 0.5 * gamma_0 # shape [B, 1]
        log_sigma_x = 0.5 * gamma_0.view(batch_size) # shape [B]

        # degrees_of_freedom_x is [B,], log_sigma_x is [B,]
        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    '''
    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        #x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        #h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        #h_int = torch.round(h_int).long() * node_mask
        h_cat = torch.zeros(0).to(z0.device)
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h
    '''
    def sample_p_xh_given_z0(self, z0_x, one_hot, molecule_mask, atom_mask, atom_edge_mask, context, fix_noise=False):
        # z0_x shape [B, N, 3, 3]
        zeros = torch.zeros(size=(z0_x.size(0), 1), device=z0_x.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2) = exp(-0.5 * gamma_0)
        sigma_x_val = torch.exp(-0.5 * gamma_0).squeeze() # [B,]
        sigma_x = self.inflate_batch_array(sigma_x_val, z0_x) #[B, 1, 1, 1]

        # Predict velocity v_0 using dynamics
        zeros = torch.zeros(size=(z0_x.size(0), 1), device=z0_x.device)
        v_0_pred = self.phi(z0_x, zeros, one_hot, molecule_mask, atom_mask, atom_edge_mask, context) # vel [B, N, 3, 3]
        # Convert velocity prediction to epsilon prediction
        alpha_0 = self.alpha(gamma_0, z0_x)
        sigma_0 = self.sigma(gamma_0, z0_x)
        eps_0_pred = self.predict_eps_from_vel(v_0_pred, z0_x, alpha_0, sigma_0)

        # Compute mu for p(x | z0) = N(mu, sigma_x)
        # mu = x_pred = (z0 - sigma_0 * eps_0_pred) / alpha_0
        mu_x = (z0_x - sigma_0 * eps_0_pred) / alpha_0

        # Sample x given mu and sigma_x
        x = self.sample_normal(mu=mu_x, sigma=sigma_x, atom_mask=atom_mask, fix_noise=fix_noise)

        # h (features) are fixed atom types, not sampled. Return None or fixed types.
        return x, None # Return positions only, features are determined by structure

    '''
    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps
    '''
    def sample_normal(self, mu, sigma, atom_mask, fix_noise=False):
        # mu/sigma shape [B, N, 3, 3] or sigma [B, 1, 1, 1]
        bs, n_mol, _, _ = mu.shape
        n_atoms_total = n_mol * self.atoms_per_molecule

        # Sample atom-level noise and center it
        eps_atoms = utils.sample_gaussian_with_mask(
            size=(bs, n_atoms_total, self.n_dims), device=mu.device, node_mask=atom_mask.view(bs, n_atoms_total, 1)
        )
        # Center atom noise across the system using atom_mask
        eps_atoms_centered = utils.remove_mean_with_mask(eps_atoms, atom_mask.view(bs, n_atoms_total, 1))

        # Reshape centered noise to molecule format
        eps = eps_atoms_centered.view(bs, n_mol, self.atoms_per_molecule, self.n_dims)

        if fix_noise:
             eps = eps[0:1].repeat(bs, 1, 1, 1) # Use noise from first sample

        return mu + sigma * eps # Apply sigma (potentially broadcasted)

    '''
    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

        # Compute delta indicator masks.
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z
    '''
    def log_pxh_given_z0_without_constants(
             self, x, # Target positions [B, N, 3, 3]
             z_t, # Sampled state z_0 [B, N, 3, 3]
             gamma_0, # Gamma at t=0 [B, 1]
             eps, # Noise used to sample z_0 [B, N, 3, 3]
             net_out_vel, # Dynamics output (velocity) [B, N, 3, 3]
             atom_mask, # Atom mask [B, max_n_atoms]
             epsilon=1e-10):

        # Convert predicted velocity to predicted epsilon
        alpha_0 = self.alpha(gamma_0, z_t)
        sigma_0 = self.sigma(gamma_0, z_t)
        net_out_eps = self.predict_eps_from_vel(net_out_vel, z_t, alpha_0, sigma_0)

        # Log prob is based on N(x | mu, sigma_x)
        # sigma_x = sqrt(sigma_0^2 / alpha_0^2) = exp(-0.5 * gamma_0)
        # mu = (z_t - sigma_0 * net_out_eps) / alpha_0
        # log p(x|z0) = -0.5 * ||x - mu||^2 / sigma_x^2 - const

        # This is complex. The VLB term derivation relies on parameterizing the score ~ eps.
        # Let's use the epsilon-based loss term directly as in DDPM L_0.
        # L_0 = E_q(z0|x) [ -log p(x|z0) ]
        #   = E_eps [ 0.5 * || eps - eps_pred(alpha_0*x + sigma_0*eps, 0) ||^2 / sigma_x^2 ] ??? No.

        # Use the standard L_simple term formulation for L_0: MSE between eps and eps_pred(z0)
        # This requires predicting eps, not velocity.
        # Let's assume net_out_vel *is* the epsilon prediction for L0 calculation simplicity.

        # Compute MSE between actual noise 'eps' and predicted noise 'net_out_eps'
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_out_eps, gamma_0, eps)

        # No feature part (log_p_h_given_z)

        return log_p_x_given_z_without_constants

    '''
    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        if torch.isnan(eps).any():
            print(f'x.size(0): {x.size(0)}, x.size(1): {x.size(1)}')
            print("Found NaNs!!!")
            print(f'NaNs found in eps at indices: {torch.isnan(eps).nonzero()}')
            print(f'xh at NaNs indices: {xh[torch.isnan(eps).nonzero(as_tuple=True)]}')
            print(f"xh: {xh}")
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}
    '''
    def compute_loss(self, x, one_hot, atom_mask, molecule_mask, edge_mask, context, t0_always):
        bs, n_mol, _, _ = x.shape
        n_atoms_total = n_mol * self.atoms_per_molecule

        if t0_always: lowest_t = 1
        else: lowest_t = 0

        # Sample timestep t
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()

        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma, alpha, sigma
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample atom-level noise eps ~ N(0, I), center it, reshape to [B, N, 3, 3]
        eps_atoms = utils.sample_gaussian_with_mask(
            size=(bs, n_atoms_total, self.n_dims), device=x.device, node_mask=atom_mask.view(bs, n_atoms_total, 1)
        )
        eps_atoms_centered = utils.remove_mean_with_mask(eps_atoms, atom_mask.view(bs, n_atoms_total, 1))
        eps = eps_atoms_centered.view(bs, n_mol, self.atoms_per_molecule, self.n_dims) # [B, N, 3, 3]

        # Sample z_t = alpha_t * x + sigma_t * eps
        z_t = alpha_t * x + sigma_t * eps # [B, N, 3, 3]

        # --- Prepare data for dynamics model ---
        net_out_vel = self.phi(z_t, t, one_hot, molecule_mask, atom_mask, edge_mask, context) # Returns predicted velocity [B, N, 3, 3]

        # Convert predicted velocity to predicted epsilon
        net_out_eps = self.predict_eps_from_vel(net_out_vel, z_t, alpha_t, sigma_t)

        # Compute error (MSE between actual eps and predicted eps)
        error = self.compute_error(net_out_eps, gamma_t, eps) # [B,]

        # --- Loss Term Calculation ---
        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error) # L2 loss, no SNR weighting
        else: # VLB loss
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze() # [B,]
            if SNR_weight.dim() == 0: SNR_weight = SNR_weight.unsqueeze(0) # Handle batch size 1

        loss_t_larger_than_zero = 0.5 * SNR_weight * error # [B,]

        # Constants term (log p(x|z0))
        neg_log_constants = -self.log_constants_p_x_given_z0(x, atom_mask) #[B,]
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # KL divergence prior term
        kl_prior = self.kl_prior(x, atom_mask) # [B,]

        # --- Combine terms ---
        if t0_always: # VLB calculation with separate L0
            loss_t = loss_t_larger_than_zero
            num_terms = self.T
            estimator_loss_terms = num_terms * loss_t

            # Calculate L0 term
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0
            eps_0_atoms = utils.sample_gaussian_with_mask(
                size=(bs, n_atoms_total, self.n_dims), device=x.device, node_mask=atom_mask.view(bs, n_atoms_total, 1)
            )
            eps_0_atoms_centered = utils.remove_mean_with_mask(eps_0_atoms, atom_mask.view(bs, n_atoms_total, 1))
            eps_0 = eps_0_atoms_centered.view(bs, n_mol, self.atoms_per_molecule, self.n_dims) # [B, N, 3, 3]
            z_0 = alpha_0 * x + sigma_0 * eps_0

            # Predict velocity/eps at t=0
            net_out_vel_0 = self.phi(z_0, t_zeros, one_hot, molecule_mask, atom_mask, edge_mask, context)
            net_out_eps_0 = self.predict_eps_from_vel(net_out_vel_0, z_0, alpha_0, sigma_0)

            # Calculate L0 using log_pxh_given_z0
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                 x, z_0, gamma_0, eps_0, net_out_vel_0, atom_mask) # Pass vel here

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0
        else: # Simpler calculation (L_simple or VLB combined)
            # Calculate L0 term (using z_t and eps at the sampled t, but gamma_t)
             loss_term_0 = -self.log_pxh_given_z0_without_constants(
                  x, z_t, gamma_t, eps, net_out_vel, atom_mask) # Pass vel here

             t_is_not_zero = 1 - t_is_zero # [B, 1]
             loss_t = loss_term_0 * t_is_zero.squeeze() + \
                      t_is_not_zero.squeeze() * loss_t_larger_than_zero # [B,]

             if self.training and self.loss_type == 'l2':
                 estimator_loss_terms = loss_t
             else: # VLB
                 num_terms = self.T + 1
                 estimator_loss_terms = num_terms * loss_t

             loss = kl_prior + estimator_loss_terms + neg_log_constants

        return loss, {'t': t_int.squeeze(), 'loss_t': loss_t.squeeze(),
                       'error': error.squeeze(), 'kl_prior': kl_prior.squeeze(),
                       'neg_log_constants': neg_log_constants.squeeze()}

    '''
    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh
    '''
    def forward(self, x, h, node_mask, edge_mask, context=None):
        # x: [B, N, 3, 3], h: {'categorical': [B, N*3, n_types], ...},
        # node_mask: atom_mask [B, N*3], edge_mask: atom_edge_mask [B, N*3, N*3]
        # molecule_mask: [B, N] - passed implicitly via atom_mask? Need to reconstruct if needed.
        atom_mask = node_mask # Rename for clarity
        atom_edge_mask = edge_mask
        one_hot = h['categorical'] # Atom features [B, N*3, n_types]

        bs, max_n_atoms, _ = one_hot.shape
        max_n_mol = max_n_atoms // self.atoms_per_molecule
        # Assuming atom_mask [B, M] has 1s for real atoms, 0s for padding
        molecule_mask = atom_mask.view(bs, max_n_mol, self.atoms_per_molecule)[:, :, 0] # [B, N]

        # Normalize coordinates
        x_norm, delta_log_px = self.normalize(x, atom_mask)

        # Reset delta_log_px if using L2 loss
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            loss, loss_dict = self.compute_loss(
                x_norm, one_hot, atom_mask, molecule_mask, atom_edge_mask, context, t0_always=False)
        else: # Evaluation uses t0_always=True for lower variance VLB
            loss, loss_dict = self.compute_loss(
                x_norm, one_hot, atom_mask, molecule_mask, atom_edge_mask, context, t0_always=True)

        neg_log_pxh = loss # Loss is the negative log-likelihood estimate

        # Correct for normalization volume change
        neg_log_pxh = neg_log_pxh - delta_log_px

        # Average over batch dimension for final loss value
        return torch.mean(neg_log_pxh)

    '''
    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        return zs
    '''
    def sample_p_zs_given_zt(self, s, t, zt_x, one_hot, molecule_mask, atom_mask, atom_edge_mask, context, fix_noise=False):
        # zt_x: [B, N, 3, 3]
        gamma_s = self.gamma(s) # [B, 1]
        gamma_t = self.gamma(t) # [B, 1]

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_x) # Shapes like [B, 1, 1, 1]

        sigma_s = self.sigma(gamma_s, target_tensor=zt_x)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_x)

        # Predict velocity v_t
        v_t_pred = self.phi(zt_x, t, one_hot, molecule_mask, atom_mask, atom_edge_mask, context) # vel [B, N, 3, 3]
        # Convert velocity prediction to epsilon prediction
        eps_t_pred = self.predict_eps_from_vel(v_t_pred, zt_x, alpha_t_given_s * self.alpha(gamma_s, zt_x), sigma_t) # Use alpha_t = alpha_t_given_s * alpha_s

        # Compute mu for p(zs | zt). mu = zt / alpha_t_given_s - sigma_t^2/alpha_t_given_s/sigma_t * eps_t
        mu = zt_x / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t_pred

        # Compute sigma for p(zs | zt). sigma = sigma_t_given_s * sigma_s / sigma_t
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs
        zs = self.sample_normal(mu, sigma, atom_mask, fix_noise) # Sample and center noise inside

        # No feature sampling needed

        return zs # [B, N, 3, 3]

    '''
    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z
    '''
    def sample_combined_position_feature_noise(self, n_samples, n_nodes_mol, atom_mask):
        # n_nodes_mol is the max number of molecules (max_N)
        # atom_mask is [B, max_n_atoms]
        bs = n_samples
        max_n_atoms = n_nodes_mol * self.atoms_per_molecule
        atom_mask_reshaped = atom_mask.view(bs, max_n_atoms, 1)

        # Sample atom-level noise and center it
        z_x_atoms = utils.sample_gaussian_with_mask(
            size=(bs, max_n_atoms, self.n_dims), device=atom_mask.device, node_mask=atom_mask_reshaped
        )
        z_x_atoms_centered = utils.remove_mean_with_mask(z_x_atoms, atom_mask_reshaped)

        # Reshape centered noise to molecule format [B, N, 3, 3]
        z_x = z_x_atoms_centered.view(bs, n_nodes_mol, self.atoms_per_molecule, self.n_dims)

        # No feature noise needed
        return z_x

    '''
    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h
    '''
    @torch.no_grad()
    def sample(self, n_samples, n_nodes_mol, molecule_mask, atom_mask, atom_edge_mask, one_hot, context, fix_noise=False):
        # n_nodes_mol = max number of molecules in the batch
        # molecule_mask [B, max_N], atom_mask [B, max_N*3], atom_edge_mask [B, max_N*3, max_N*3]

        # Sample initial noise z_T in the correct shape and centered
        z_x = self.sample_combined_position_feature_noise(n_samples, n_nodes_mol, atom_mask)

        # Iteratively sample p(z_s | z_t)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z_x.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            z_x = self.sample_p_zs_given_zt(
                s_norm, t_norm, z_x,
                one_hot, molecule_mask, atom_mask, atom_edge_mask,
                context, fix_noise=fix_noise
            )


        # Finally sample p(x | z_0).
        x, _ = self.sample_p_xh_given_z0(
            z_x,
            one_hot, molecule_mask, atom_mask, atom_edge_mask,
            context, fix_noise=fix_noise
        )

        # Final centering check (optional but good practice)
        x_atoms_final = x.view(n_samples, -1, self.n_dims)
        atom_mask_final = atom_mask.view(n_samples, -1, 1)
        com_drift = torch.sum(x_atoms_final * atom_mask_final, dim=1) / torch.sum(atom_mask_final, dim=1).clamp(min=1)
        max_cog_err = com_drift.abs().max().item()
        if max_cog_err > 1e-2: # Adjust threshold as needed
            print(f'Warning: Final COM drift {max_cog_err:.3f}. Re-centering.')
            x_atoms_final_centered = utils.remove_mean_with_mask(x_atoms_final, atom_mask_final)
            x = x_atoms_final_centered.view(n_samples, n_nodes_mol, self.atoms_per_molecule, self.n_dims)

        # Unnormalize the final positions
        x_unnorm = self.unnormalize(x)

        # Features 'h' are determined by the structure (O, H, H), not sampled.
        # We might need to return the fixed atom types alongside x_unnorm.
        num_atom_types = self.in_node_nf
        fixed_atom_types_flat = torch.tensor([0, 1, 1] * n_nodes_mol, device=x.device).long() # [max_N*3]
        fixed_one_hot = F.one_hot(fixed_atom_types_flat, num_classes=num_atom_types).float() # [max_N*3, num_types]
        # Apply atom mask to one_hot
        fixed_one_hot = fixed_one_hot.unsqueeze(0).repeat(n_samples, 1, 1) # [B, max_N*3, num_types]
        fixed_one_hot = fixed_one_hot * atom_mask.unsqueeze(-1)

        h_final = {'categorical': one_hot, 'integer': torch.zeros(0)} # Placeholder for integer

        return x_unnorm, h_final # Return unnormalized positions and fixed features

    '''
    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context)

            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat
    '''
    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes_mol, molecule_mask, atom_mask, atom_edge_mask, context, keep_frames=None):
        # n_nodes_mol = max number of molecules

        z_x = self.sample_combined_position_feature_noise(n_samples, n_nodes_mol, atom_mask)

        if keep_frames is None: keep_frames = self.T
        else: assert keep_frames <= self.T
        chain_x = torch.zeros((keep_frames,) + z_x.size(), device=z_x.device) # Store positions

        # Iterative sampling
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z_x.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            z_x = self.sample_p_zs_given_zt(
                s_norm, t_norm, z_x,
                one_hot, molecule_mask, atom_mask, atom_edge_mask,
                context, fix_noise=False
            )

            # Store frame if needed (unnormalized)
            write_index = (s * keep_frames) // self.T
            if write_index < keep_frames: # Ensure index is valid
                 chain_x[write_index] = self.unnormalize(z_x)

        # Final sample p(x | z_0)
        x, _ = self.sample_p_xh_given_z0(
            z_x,
            one_hot, molecule_mask, atom_mask, atom_edge_mask,
            context, fix_noise=False
        )
        x_unnorm = self.unnormalize(x)

        # Overwrite the first frame (t=0) with the final sample
        if 0 < keep_frames:
            chain_x[0] = x_unnorm

        # We don't store features in the chain as they are fixed.
        return chain_x # Return only the position chain [keep_frames, B, N, 3, 3]

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info
