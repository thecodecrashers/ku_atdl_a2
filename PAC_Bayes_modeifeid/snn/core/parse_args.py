import argparse
import os

from snn.core.cnn import CNN
from snn.core.data_fn import load_binary_mnist, load_cifar_data, load_mnist_data
from snn.core.fc import FC


class BasicParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("model", help="Model to be used (FC or CNN)", type=str)
        self.parser.add_argument("--layers", help="Layer architecture (can only be used with FC)", type=int, nargs='+',
                                 required=False, default=[784, 600, 10])
        self.parser.add_argument("--sgd_epochs", help="Number of epochs", type=int, required=False, default=1)
        # Added a separate update limit so the short-SGD ablation can perform
        # one mini-batch update without confusing that with one full epoch.
        self.parser.add_argument("--sgd_steps", help="Limit SGD to this many mini-batch updates", type=int, default=None)
        self.parser.add_argument("--seed", help="Random seed", type=int, required=False, default=11)
        self.parser.add_argument("--binary", action='store_true')
        self.parser.add_argument("--overwrite", action='store_true')
        self.parser.add_argument("--device", help="Torch device (default: cuda if available, else cpu)", type=str, required=False, default=None)
        # Per-run directories isolate checkpoints, histories, and summaries.
        # The experiment launchers require fresh roots to preserve prior results.
        self.parser.add_argument("--output_dir", help="Output directory", type=str, required=False, default=None)
        self.parser.add_argument("--run_name", help="Run name", type=str, required=False, default=None)
        self.parser.add_argument("--random_labels", help="Randomize training labels", action="store_true")

    def get_args(self, args):
        return {"model": args.model, "layers": args.layers, "sgd_epochs": args.sgd_epochs, "sgd_steps": args.sgd_steps, "seed": args.seed, "binary": args.binary, "overwrite": args.overwrite, "device": args.device, "output_dir": args.output_dir, "run_name": args.run_name, "random_labels": getattr(args, "random_labels", False)}

    def parse(self):
        args = self.parser.parse_args()
        print("Input args: ", args)
        return self.get_args(args)


class CompleteParser(BasicParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("--pacb_epochs", help="Number of PAC-Bayes epochs", type=int, required=False, default=1)
        self.parser.add_argument("--lr", help="Learning rate for PACB optimization", type=float, required=False,
                            default=0.001)
        self.parser.add_argument("--drop_lr", help="Number of epochs before dropping learning rate", type=float,
                                 required=False, default=20)
        self.parser.add_argument("--lr_factor", help="Factor by which the learning rate is dropped", type=float,
                                 required=False, default=0.05)
        # Upstream uses type=bool: the string "False" is nonempty, so
        # --trainw False incorrectly becomes True. BooleanOptionalAction exposes
        # --trainw / --no-trainw and reliably controls posterior mean training.
        # Freezing the means does not freeze posterior/prior variance parameters.
        self.parser.add_argument("--trainw", help="Train posterior weights during PAC-Bayes optimization", action=argparse.BooleanOptionalAction, default=True)
        # Upstream run_pacb.py hardcodes one final draw. Expose N independently
        # of the training budget: 1 is quick but gives a very loose MC correction;
        # 1000 is a practical evaluation, and the paper reports N=150000.
        self.parser.add_argument("--snn_samples", help="Independent posterior draws for final evaluation; 1 is fast but gives a loose corrected bound", type=int, required=False, default=1)
        # This controls optional epoch diagnostics only. The final independent
        # posterior evaluation still runs when the interval is zero.
        self.parser.add_argument("--eval_interval", help="Evaluate training accuracy every N PAC-Bayes epochs; 0 evaluates only after training", type=int, default=1)

    def get_args(self, args):
        pacb_args = {"pacb_epochs": args.pacb_epochs, "lr": args.lr, "drop_lr": args.drop_lr, "lr_factor": args.lr_factor, "trainw": args.trainw, "snn_samples": getattr(args, "snn_samples", 1), "eval_interval": args.eval_interval}
        complete_args = super().get_args(args)
        complete_args.update(pacb_args)
        return complete_args


class Interpreter(object):
    def __init__(self, input_args):
        self.input_args = input_args

    def interpret(self, initial_weights=None):
        norm_name = self.input_args["model"].upper()
        layers, epochs, seed = self.input_args["layers"], self.input_args["sgd_epochs"], self.input_args["seed"]
        device = self.input_args.get("device")
        scopes_list = ["hidden" + str(i+1) for i in range(len(layers))]
        scopes_list.append("output")
        if norm_name == "FC":
            if self.input_args["binary"]:
                path = os.path.join("binary_mnist", "{}_layers{}_epochs{}_seed{}.pickle".format(norm_name, layers,
                                                                                                epochs, seed))
                (trainX, trainY), (testX, testY) = load_binary_mnist(random_labels=self.input_args.get("random_labels", False), seed=seed)
                model = FC(trainX, trainY, layers=[784] + layers + [1], scopes_list=scopes_list,
                           seed=seed, initial_weights=initial_weights, device=device)
            else:
                path = os.path.join("mnist", "{}_layers{}_epochs{}_seed{}.pickle".format(norm_name, layers, epochs,
                                                                                         seed))
                (trainX, trainY), (testX, testY) = load_mnist_data()
                model = FC(trainX, trainY, layers=[784] + layers + [10], scopes_list=scopes_list,
                           seed=seed, initial_weights=initial_weights, device=device)
        elif norm_name == "CNN":
            path = os.path.join("cifar", "{}_epochs{}_seed{}.pickle".format(norm_name, epochs, seed))
            (trainX, trainY), (testX, testY) = load_cifar_data()
            model = CNN(trainX, trainY, seed=seed, initial_weights=initial_weights, device=device)
        else:
            raise NotImplementedError("The model '{}' has not been implemented".format(self.input_args["model"]))
        return model, (testX, testY), path
