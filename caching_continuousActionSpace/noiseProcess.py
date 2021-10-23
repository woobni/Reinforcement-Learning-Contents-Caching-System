import numpy as np


class OrnsteinUhlenbeckProcess(object):
    """Ornstein-Uhlenbeck process.
    formula：ou = θ * (μ - x) + σ * w
    Arguments:
        x: action value.
        mu: μ, mean fo values.
        theta: θ, rate the variable reverts towards to the mean.
        sigma：σ, degree of volatility of the process.
    Returns:
        OU value
    """
    def __init__(self, theta=0.15, mu=0, sigma=0.2, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)  # -0.002. no noise after 100 steps?
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, action_, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.theta * (self.mu - action_) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        # x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        # self.x0 = x
        return x


