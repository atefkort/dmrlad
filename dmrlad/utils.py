import torch
import torch.nn.functional as F


def product_of_gaussians3D(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
    return mu, sigma_squared

def generate_gaussian(mu_sigma, latent_dim, sigma_ops="softplus", mode=None):
    """
    Generate a Gaussian distribution given a selected parametrization.
    """
    mus, sigmas = torch.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)

    if sigma_ops == 'softplus':
        # Softplus, s.t. sigma is always positive
        # sigma is assumed to be st. dev. not variance
        sigmas = F.softplus(sigmas)
    if mode == 'multiplication':
        mu, sigma = product_of_gaussians3D(mus, sigmas)
    else:
        mu = mus
        sigma = sigmas
    return torch.distributions.normal.Normal(mu, sigma)


class LinearSchedule(object):

    def __init__(self, schedule_timesteps, initial=1., final=0.):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final = final
        self.initial = initial

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial + fraction * (self.final - self.initial)