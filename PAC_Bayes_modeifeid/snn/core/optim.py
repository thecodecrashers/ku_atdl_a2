import torch
from torch.optim import Optimizer


class TFRMSprop(Optimizer):
    """
    RMSProp with the exact update rule of the original `tf.train.RMSPropOptimizer` (TensorFlow 1.x):

        ms  <- decay * ms + (1 - decay) * g^2            (ms is initialised to ONES)
        var <- var - lr * g / sqrt(ms + eps)

    `torch.optim.RMSprop` differs (alpha=0.99, ms initialised to zero, eps added outside the square root), so this
    small optimizer is used to reproduce the original optimisation dynamics.
    """

    def __init__(self, params, lr=0.001, decay=0.9, eps=1e-10):
        super().__init__(params, dict(lr=lr, decay=decay, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "ms" not in state:
                    # TensorFlow 1 RMSProp starts its squared-gradient accumulator
                    # at one, not zero. Matching this matters especially for the
                    # early PAC steps and the newly created dropped-LR optimizer.
                    state["ms"] = torch.ones_like(p)
                ms = state["ms"]
                ms.mul_(group["decay"]).addcmul_(p.grad, p.grad, value=1.0 - group["decay"])
                # Keep epsilon inside sqrt(ms+eps), as upstream TF1 does.
                # Substituting default torch.optim.RMSprop would change this
                # denominator and the accumulator/decay defaults, so the port
                # uses this explicit update rather than only matching the name.
                p.addcdiv_(p.grad, (ms + group["eps"]).sqrt(), value=-group["lr"])
        return loss
