from snn.core.mlp_fn import (multilayer_perceptron, multilayer_perceptron_params, MLP_withnoise)
from snn.core.network import Network


class FC(Network):
    """ A Fully Connected neural network learning model """

    def __init__(self, X, Y, logging=True, layers=[784, 600, 10], scopes_list=['hidden1', 'output'],
                 seed=11, initial_weights=None, device=None):
        Network.__init__(self, X, Y, logging, layers, scopes_list, seed, device)

        # PyTorch replaces the original TensorFlow graph/session variables with
        # explicit leaf tensors stored in self.params. The inherited weight
        # getter returns detached NumPy snapshots for serialization; forward
        # evaluation and KL training use the live tensors, not those snapshots.
        self.model = multilayer_perceptron  # The base network
        self.model_with_noise = MLP_withnoise
        if initial_weights is None:
            self._set_params(multilayer_perceptron_params(self.layers))
        else:  # Else, have to initialize according to the passed in weights
            self._set_params(initial_weights)
        return

    def count_N_params(self):
        """
        For computing VC dimension bound with boundVCdim()
        """
        N = 0
        par_shapes = []
        for (n_in, n_out) in zip(self.layers[:-1], self.layers[1:]):
            N += n_in * n_out
            N += n_out
            par_shapes.append([n_in,n_out])
            par_shapes.append([n_out])
        return N, par_shapes

    def load_model_weights(self, params_mean_values):
        self._set_params(params_mean_values)
        return self.params
