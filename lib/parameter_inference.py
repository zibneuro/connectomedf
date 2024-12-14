import numpy as np
import torch

from sbi.inference import SNPE
from sbi.inference import DirectPosterior
from sbi.utils import BoxUniform
from sbi.utils import posterior_nn
from sbi import analysis as analysis

class ParameterDomain():
    def __init__(self):
        self.min_ranges = []
        self.max_ranges = []
        self.keys = []
        self.labels = []
    
    def add_parameter(self, key, min, max, label=None):  
        self.keys.append(key)
        self.min_ranges.append(min)
        self.max_ranges.append(max)
        if(label is None):
            parameter_idx = len(self.keys)
            self.labels.append(r"$\alpha_{{{}}}$".format(parameter_idx))
        else:
            self.labels.append(label)

    def get_prior(self):
        return BoxUniform(low=torch.tensor(self.min_ranges), high=torch.tensor(self.max_ranges))
    
    def get_limits(self):
        return [[self.min_ranges[i], self.max_ranges[i]] for i in range(len(self.min_ranges))]
    
    def get_parameter_column_index(self, key):
        return self.keys.index(key)


class ParameterInference():
    def __init__(self, parameter_domain, num_posterior_samples = 1000, max_epochs = 100):
        self.num_posterior_samples = num_posterior_samples  
        self.max_epochs = max_epochs
        self.domain = parameter_domain

    def sample_parameters(self, num_samples):
        self.prior = self.domain.get_prior()
        self.parameters = self.prior.sample((num_samples,))
        return self.parameters.numpy()
    
    def infer_parameters(self, x_simulated, x_observed):
        x_simulated = torch.tensor(x_simulated, dtype=torch.float32)
        x_observed = torch.tensor(x_observed, dtype=torch.float32)

        assert self.prior is not None
        assert self.parameters is not None

        inferer = SNPE(
            prior=self.prior,
            show_progress_bars=True,
            density_estimator=posterior_nn(model="mdn"),        
        )
        inferer.append_simulations(self.parameters, x_simulated)
        de = inferer.train(max_num_epochs=self.max_epochs)

        self.posterior = DirectPosterior(
            posterior_estimator=de,
            prior=self.prior,
            x_shape=(1, x_simulated.shape[1])       
        )

        self.posterior.set_default_x(x_observed)
        self.samples_posterior = self.posterior.sample((self.num_posterior_samples,))
        self.prob_posterior_samples = self.posterior.log_prob(self.samples_posterior).numpy()

        indices_sorted = np.flip(np.argsort(self.prob_posterior_samples))
        return self.samples_posterior.numpy()[indices_sorted[0], :]

    def plot_posterior(self, figsize=(10,10), limits=None):
        assert self.samples_posterior is not None

        if(limits is None):
            limits = self.domain.get_limits()

        fig, axes = analysis.pairplot(self.samples_posterior, limits=limits, figsize=figsize, labels=self.domain.labels)
        return fig, axes