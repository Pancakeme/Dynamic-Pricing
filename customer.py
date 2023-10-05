import numpy as np

class CustomerSimulator(object):
    def __init__(self, params, factor=80):
        self.params = np.asarray(params)
        self.num_comps = 1
        self.factor = factor

    def get_lambda(self, length, a, p, r):
        features = np.array([
            1,
            (1 + ((1 if p < a else 0) + (1 if p <= a else 0)) / 2),
            a - p,
            self.num_comps,
            (a + p) / 2,
            a - r
        ])
        return (length * np.exp(np.dot(self.params, features)) / (
                1 + np.exp(np.dot(self.params, features)))) * self.factor

    def __call__(self, length, own_price, comp_price, ref_price):
        return np.random.poisson(self.get_lambda(length, own_price, comp_price, ref_price))
    
DEFAULT_PARAMS = [-3.89, -0.56, -0.01, 0.07, -0.03, 0]
DEFAULT_PARAMS_REF = [-3.89, -0.56, -0.01, 0.07, -0.03, -0.01]
DEFAULT_SIM = CustomerSimulator(DEFAULT_PARAMS)