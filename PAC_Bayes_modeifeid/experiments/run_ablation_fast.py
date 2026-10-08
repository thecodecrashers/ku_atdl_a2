"""Run random initialization, SGD initialization, and frozen SGD weights concurrently."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import re
import subprocess
import sys
import threading
import time
import uuid


PKG_ROOT = Path(__file__).resolve().parents[1]
SGD_SCRIPT = PKG_ROOT / "snn" / "experiments" / "run_sgd.py"
PACB_SCRIPT = PKG_ROOT / "snn" / "experiments" / "run_pacb.py"
# New experiments relative to upstream, all with the T600 architecture:
# random_init: no SGD; optimize means and variances during stage two.
# sgd_init: one mini-batch SGD update by default; then optimize means/variances.
# no_trainw: 20 SGD epochs by default; freeze all means during stage two and
# optimize only posterior standard deviations and the prior standard deviation.
CONDITIONS = ("random_init", "sgd_init", "no_trainw")


def format_duration(seconds):
    if seconds is None:
        return "estimating"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class RunProgress:
    """Report progress from child logs without evaluating the training set."""

    def __init__(self, interval=30):
        self.interval = interval
        self.lock = threading.Lock()
        self.states = {}

    def start(self, name, stage, total, unit):
        now = time.monotonic()
        with self.lock:
            self.states[name] = {"stage": stage, "done": 0, "total": total, "unit": unit,
                                 "started": now, "updated": now, "printed": now,
                                 "samples": deque(maxlen=20), "detail": ""}
            print(f"[{name}] {stage} started: {total} {unit}", flush=True)

    def _print(self, name, state, now):
        # ETA is estimated from recent observed updates of the current stage.
        # It is not a guaranteed finish time or an ETA for the complete suite;
        # final Monte Carlo evaluation is reported as a separate stage.
        samples = state["samples"]
        remaining = None
        if state["done"] == state["total"]:
            remaining = 0
        elif len(samples) >= 2:
            amount = samples[-1][0] - samples[0][0]
            if amount > 0:
                remaining = ((state["total"] - state["done"])
                             * (samples[-1][1] - samples[0][1]) / amount)
                remaining = max(0, remaining - (now - samples[-1][1]))
        percent = 100 * state["done"] / state["total"] if state["total"] else 100
        detail = f" | {state['detail']}" if state["detail"] else ""
        stale = f" | last update {round(now-state['updated'])}s ago" if now-state["updated"] > 2*self.interval else ""
        print(f"[{name}] {state['stage']}: {state['done']}/{state['total']} {state['unit']} "
              f"({percent:.1f}%) | elapsed {format_duration(now-state['started'])} "
              f"| stage ETA {format_duration(remaining)}{detail}{stale}", flush=True)
        state["printed"] = now

    def update(self, name, done, total=None, detail=""):
        now = time.monotonic()
        with self.lock:
            state = self.states[name]
            first = not state["samples"]
            if total is not None:
                state["total"] = total
            if first or done > state["done"]:
                state["samples"].append((done, now))
            state.update(done=done, detail=detail, updated=now)
            if first or done == state["total"] or now-state["printed"] >= self.interval:
                self._print(name, state, now)

    def finish(self, name):
        with self.lock:
            self.states.pop(name, None)

    def heartbeat(self):
        now = time.monotonic()
        with self.lock:
            for name, state in self.states.items():
                if now-state["printed"] >= self.interval:
                    self._print(name, state, now)


def run_stage(name, stage, command, env, log, progress, total, unit):
    """Stream output to disk and expose selected progress messages in the terminal."""
    # Progress is parsed from the child's ordinary stdout, so heartbeat and
    # ETA reporting do not trigger extra model evaluations or random draws.
    # Flush the complete log while the terminal displays selected status lines.
    progress.start(name, stage, total, unit)
    tail = deque(maxlen=12)
    with subprocess.Popen(command, cwd=str(PKG_ROOT), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as child:
        for line in child.stdout:
            log.write(line)
            log.flush()
            tail.append(line.rstrip())
            if stage == "SGD":
                iteration = re.search(r"SGD iteration:\s*(\d+)/(\d+)", line)
                if iteration:
                    done = int(iteration.group(1))
                    progress.update(name, done, int(iteration.group(2)),
                                    detail=f"working on epoch {done // 550 + 1}")
            else:
                iteration = re.search(r"Iteration:\s*(\d+)/(\d+)", line)
                if iteration:
                    epoch = re.match(r"Epoch:(\d+)", line)
                    kl = re.search(r"KL div:\s*([-+\d.eE]+)", line)
                    detail = f"epoch {int(epoch.group(1))}" if epoch else ""
                    if kl:
                        detail += f", KL={kl.group(1)}"
                    progress.update(name, int(iteration.group(1)), int(iteration.group(2)), detail)
                evaluation_start = re.match(r"Final posterior evaluation started: (\d+) samples", line)
                if evaluation_start:
                    progress.start(name, "Final evaluation", int(evaluation_start.group(1)), "samples")
                    progress.update(name, 0)
                evaluation = re.match(r"Final posterior evaluation: (\d+)/(\d+)", line)
                if evaluation:
                    progress.update(name, int(evaluation.group(1)), int(evaluation.group(2)))
                if line.startswith(("PAC bound error:", "Train error upper bound:")):
                    print(f"[{name}] {line.strip()}", flush=True)
        returncode = child.wait()
    if returncode == 0 and stage == "SGD":
        progress.update(name, total)
    progress.finish(name)
    if returncode:
        print(f"[{name}] {stage} exited with code {returncode}. Last log lines:", flush=True)
        print("\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(returncode, command)
    print(f"[{name}] {stage} completed.", flush=True)


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def run_job(name, args, root, progress=None):
    """Use a separate process so PyTorch and random-number states are isolated."""
    out_dir = root / name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "run.log"
    progress = progress or RunProgress(getattr(args, "progress_interval", 30))
    random_init = name == "random_init"
    frozen_weights = name == "no_trainw"
    # Every job starts from a fresh initialization, then prepares its own
    # checkpoint. None of these jobs reuses baseline T600 weights or MC seeds.
    if random_init:
        sgd_epochs = 0
    elif frozen_weights:
        sgd_epochs = args.frozen_sgd_epochs
    elif args.sgd_epochs is not None:
        sgd_epochs = args.sgd_epochs
    else:
        # Binary MNIST has 550 mini-batches per epoch at batch size 100.
        sgd_epochs = max(1, (args.sgd_steps + 549) // 550)

    env = os.environ.copy()
    # Prevent the three workers from oversubscribing CPU threads.
    env["OMP_NUM_THREADS"] = str(args.threads_per_run)
    env["MKL_NUM_THREADS"] = str(args.threads_per_run)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    model_args = ["fc", "--layers", "600", "--sgd_epochs", str(sgd_epochs),
                  "--binary", "--seed", str(args.seed), "--output_dir", str(out_dir),
                  "--run_name", name]
    if args.device != "auto":
        model_args.extend(["--device", args.device])

    print(f"[{name}] Starting. Log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        sgd_args = list(model_args)
        # Allocate enough SGD epochs to accommodate the requested step budget,
        # but let --sgd_steps enforce the actual mini-batch count. With the
        # default budget, only one update runs even though one epoch is allocated.
        if name == "sgd_init" and args.sgd_steps is not None:
            sgd_args.extend(["--sgd_steps", str(args.sgd_steps)])
        sgd_total = (min(args.sgd_steps, sgd_epochs * 550)
                     if name == "sgd_init" and args.sgd_steps is not None else sgd_epochs * 550)
        run_stage(name, "SGD", [sys.executable, "-u", str(SGD_SCRIPT), *sgd_args],
                  env, log, progress, sgd_total, "updates")

        # Only no_trainw disables mean training. The other two conditions change
        # initialization, not the PAC optimizer's trainable parameter groups.
        # All three receive the same stage-two hyperparameters and MC draw count.
        pacb_args = ["--pacb_epochs", str(args.pacb_epochs), "--lr", "0.001",
                     "--drop_lr", "250", "--lr_factor", "0.1",
                     "--no-trainw" if frozen_weights else "--trainw",
                     "--snn_samples", str(args.snn_samples),
                     "--eval_interval", str(args.eval_interval)]
        run_stage(name, "PAC-Bayes", [sys.executable, "-u", str(PACB_SCRIPT), *model_args, *pacb_args],
                  env, log, progress, args.pacb_epochs * 550, "updates")
    print(f"[{name}] Finished. Results: {out_dir}", flush=True)
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None,
                        help="Initialization seed; default generates a fresh seed independent of baseline runs")
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--sgd_steps", type=positive_int,
                        help="Mini-batch updates for the SGD-initialized condition; default 1")
    warmup.add_argument("--sgd_epochs", type=positive_int,
                        help="Train this many SGD epochs instead of using a step limit")
    parser.add_argument("--frozen_sgd_epochs", type=positive_int, default=20,
                        help="SGD pretraining epochs for the frozen-weight condition; default 20")
    parser.add_argument("--pacb_epochs", type=nonnegative_int, default=1000)
    parser.add_argument("--eval_interval", type=nonnegative_int, default=0,
                        help="Training evaluation interval in epochs; default 0 evaluates only after training")
    parser.add_argument("--snn_samples", type=positive_int, default=1,
                        help="Independent posterior draws for final evaluation; 1 is fast but gives a loose corrected bound")
    parser.add_argument("--threads_per_run", type=positive_int, default=2,
                        help="CPU threads per training process")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch CUDA device")
    parser.add_argument("--output_dir", help="New output directory; must not already exist")
    parser.add_argument("--progress_interval", type=positive_int, default=30,
                        help="Seconds between terminal progress updates; default 30")
    args = parser.parse_args(argv)

    if args.seed is None:
        # Share a newly generated training seed across these three conditions,
        # so their starting random weights match. It is independent of the
        # baseline seed; final MC evaluation also uses a separate fresh RNG.
        args.seed = secrets.randbelow(2**31)
    if args.sgd_steps is None and args.sgd_epochs is None:
        args.sgd_steps = 1

    # Claim a new directory before launching any worker; never overwrite a run.
    root = (Path(args.output_dir).resolve() if args.output_dir else
            PKG_ROOT / "results" / f"ablation_fast_{datetime.now():%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}")
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists; choose a new directory: {root}")
    config = vars(args).copy()
    config["output_dir"] = str(root)
    config["conditions"] = list(CONDITIONS)
    config["train_posterior_mean"] = {name: name != "no_trainw" for name in CONDITIONS}
    config["initialization_source"] = "fresh_random_weights"
    (root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Output directory: {root}", flush=True)
    print(f"Training evaluation interval: {args.eval_interval} (0 = final evaluation only)", flush=True)
    print(f"Fresh initialization seed: {args.seed}", flush=True)
    warmup_text = (f"{args.sgd_epochs} epochs" if args.sgd_epochs is not None
                   else f"{args.sgd_steps} mini-batch updates")
    print(f"Conditions: random initialization; SGD initialization after {warmup_text}", flush=True)
    print(f"Third condition: frozen weights after {args.frozen_sgd_epochs} SGD epochs", flush=True)

    failures = []
    progress = RunProgress(args.progress_interval)
    # Three coordinator threads launch isolated Python subprocesses. Training
    # itself does not share model objects or global RNG state between threads.
    # CPU thread limits reduce oversubscription; GPU workers still share one
    # physical device, so concurrency does not guarantee a threefold speedup.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(run_job, name, args, root, progress): name
                for name in CONDITIONS}
        pending = set(jobs)
        while pending:
            completed, pending = wait(pending, timeout=args.progress_interval, return_when=FIRST_COMPLETED)
            progress.heartbeat()
            for job in completed:
                try:
                    job.result()
                except Exception as error:
                    name = jobs[job]
                    progress.finish(name)
                    failures.append(name)
                    print(f"[{name}] Failed: {error}. Check {root / name / f'seed{args.seed}' / 'run.log'}", flush=True)
    if failures:
        return 1
    print("All three ablations completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
