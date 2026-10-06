import os

ROOT = r"c:\Users\戴子寒\OneDrive - The Chinese University of Hong Kong\Desktop\硕士2block1\ATDL\assignment2\PAC_Bayes_modeifeid"

def patch_parse_args():
    path = os.path.join(ROOT, "snn/core/parse_args.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        'self.parser.add_argument("--device", help="Torch device (default: cuda if available, else cpu)", type=str,\n                                 required=False, default=None)',
        'self.parser.add_argument("--device", help="Torch device (default: cuda if available, else cpu)", type=str, required=False, default=None)\n        self.parser.add_argument("--output_dir", help="Output directory", type=str, required=False, default=None)\n        self.parser.add_argument("--run_name", help="Run name", type=str, required=False, default=None)\n        self.parser.add_argument("--random_labels", help="Randomize training labels", action="store_true")'
    )
    content = content.replace(
        'return {"model": args.model, "layers": args.layers, "sgd_epochs": args.sgd_epochs, "seed": args.seed,\n                "binary": args.binary, "overwrite": args.overwrite, "device": args.device}',
        'return {"model": args.model, "layers": args.layers, "sgd_epochs": args.sgd_epochs, "seed": args.seed, "binary": args.binary, "overwrite": args.overwrite, "device": args.device, "output_dir": args.output_dir, "run_name": args.run_name, "random_labels": getattr(args, "random_labels", False)}'
    )
    content = content.replace(
        'self.parser.add_argument("--trainw", help="Train posterior weights during PAC-Bayes optimization", type=bool,\n                            required=False, default=True)',
        'self.parser.add_argument("--trainw", help="Train posterior weights during PAC-Bayes optimization", action=argparse.BooleanOptionalAction, default=True)\n        self.parser.add_argument("--snn_samples", help="Stochastic evaluation samples", type=int, required=False, default=1)'
    )
    content = content.replace(
        'pacb_args = {"pacb_epochs": args.pacb_epochs, "lr": args.lr, "drop_lr": args.drop_lr,\n                     "lr_factor": args.lr_factor, "trainw": args.trainw}',
        'pacb_args = {"pacb_epochs": args.pacb_epochs, "lr": args.lr, "drop_lr": args.drop_lr, "lr_factor": args.lr_factor, "trainw": args.trainw, "snn_samples": getattr(args, "snn_samples", 1)}'
    )
    content = content.replace(
        '(trainX, trainY), (testX, testY) = load_binary_mnist()',
        '(trainX, trainY), (testX, testY) = load_binary_mnist(random_labels=self.input_args.get("random_labels", False), seed=seed)'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def patch_data_fn():
    path = os.path.join(ROOT, "snn/core/data_fn.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        'def load_binary_mnist(data_dir = MNIST_DATA_DIR, validation_size=MNIST_VALIDATION_SIZE):',
        'def load_binary_mnist(data_dir = MNIST_DATA_DIR, validation_size=MNIST_VALIDATION_SIZE, random_labels=False, seed=11):'
    )
    content = content.replace(
        'testY = binarize_mnist_labels(testY).astype(np.float32)\n    return (trainX, trainY), (testX, testY)',
        'testY = binarize_mnist_labels(testY).astype(np.float32)\n    if random_labels:\n        rng = np.random.RandomState(seed)\n        idx = rng.permutation(len(trainY))\n        trainY = trainY[idx]\n    return (trainX, trainY), (testX, testY)'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def patch_network():
    path = os.path.join(ROOT, "snn/core/network.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        'def optimize_PACB(self, prior_weights, epochs=20, learning_rate=0.01, batch_size=100, drop_lr=10, lr_factor=0.05,\n                    save_dict=None, trainWeights=True):',
        'def optimize_PACB(self, prior_weights, epochs=20, learning_rate=0.01, batch_size=100, drop_lr=10, lr_factor=0.05,\n                    save_dict=None, trainWeights=True):\n        self.history = []'
    )
    history_code = """
            if i % (1 * Nsamples / batch_size) == 0 or (i == epochs * int(Nsamples / batch_size) - 1):
                bpac = approximate_BPAC_bound(train_accuracy_stoch, B_i)
                self.history.append({
                    "epoch": epoch,
                    "A_term": A_i,
                    "B_term": B_i,
                    "objective": cost_i,
                    "stochastic_train_accuracy": train_accuracy_stoch,
                    "stochastic_train_error": 1.0 - train_accuracy_stoch,
                    "KL": kldiv2_i / 2,
                    "PAC_bound_estimate": bpac,
                    "log_prior_std": _log_prior_std,
                    "learning_rate": lr_factor * learning_rate if lr_dropped else learning_rate
                })
"""
    content = content.replace(
        '            # Save at a frequency for every epoch, or at the last run\n            if i%(1 * Nsamples / batch_size)==0 or (i == epochs*int(Nsamples/batch_size) - 1):\n                bpac = approximate_BPAC_bound(train_accuracy_stoch, B_i)',
        history_code + '            if i%(1 * Nsamples / batch_size)==0 or (i == epochs*int(Nsamples/batch_size) - 1):\n                bpac = approximate_BPAC_bound(train_accuracy_stoch, B_i)'
    )
    content = content.replace(
        'return bpac, B_val, KL_val, self.deltaPAC',
        'return bpac, B_val, KL_val, self.deltaPAC, mean_train_accuracy, mean_test_accuracy, self.log_prior_std'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    patch_parse_args()
    patch_data_fn()
    patch_network()
    print("Core files patched.")
