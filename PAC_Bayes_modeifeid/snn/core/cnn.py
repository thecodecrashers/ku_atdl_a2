from snn.core.cnn_fn import convolutional_net, convolutional_net_params, CNN_withnoise
from snn.core.network import Network

IMAGE_SIZE = 32
NUM_CLASSES = 10


class CNN(Network):
    """ A network model """

    def __init__(self, X, Y, logging=True, layers = None, scopes_list=['conv1', 'conv2', 'local3', 'local4', 'softmax_linear'],
                 seed=11, initial_weights=None, device=None):
        if layers is None:  # only layers[-1] (the number of classes) is used by the model
            layers = [IMAGE_SIZE * IMAGE_SIZE * 3, NUM_CLASSES]
        Network.__init__(self, X, Y, logging, layers, scopes_list, seed, device) # Initialize according to the network class

        # The CNN port shares Network's live-parameter and snapshot distinction.
        # Convolution kernels keep their saved TensorFlow layout; the forward
        # helper converts layout at the operation, not in checkpoint storage.
        self.model = convolutional_net  # The base network
        self.model_with_noise = CNN_withnoise
        if initial_weights is None:
            self._set_params(convolutional_net_params(self.scopes_list))
        else:  # Else, have to initialize according to the passed in weights
            self._set_params(initial_weights)
        return

    def count_N_params(self):
        try:
            return self.N_params # Always return the saved instantiation when you can
        except AttributeError: # Else, calculate it again (should not be calculated again during the PAC Bound)
            self.N_params = Network.count_N_params(self)
            return self.N_params

    def load_model_weights(self, params_mean_values):
        self._set_params(params_mean_values)
        return self.params
