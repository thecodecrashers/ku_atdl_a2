# Retrieve the package location
import os
import snn
import inspect
package_path = os.path.dirname(inspect.getfile(snn))

import torch


def get_device(name=None):
    """ Returns the torch device to use (CUDA if available unless a device is requested explicitly). """
    if name is None or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
