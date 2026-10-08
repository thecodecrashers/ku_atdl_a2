"""Evaluate saved posteriors with paper-sized Monte Carlo sampling in a new directory."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime
import uuid


PKG_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PKG_ROOT / "results"
ABLATION_ROOT = RESULTS / "ablation_fast_1007_180347_3dbfe93d"
PREVIOUS_COMPARISON = ABLATION_ROOT / "evaluation_mc1000_34035a38" / "comparison.json"
POSTERIORS = {
    # These are existing corrected final models. This runner never invokes
    # stage-one SGD or stage-two optimization; it only reevaluates their Q.
    "T600 baseline": RESULTS / "baseline_T600_dac9a7ea159a4c0da6949cf6c87d9241" / "custom" / "T600" / "seed11",
    "random_init": ABLATION_ROOT / "random_init" / "seed1141091339",
    "sgd_init": ABLATION_ROOT / "sgd_init" / "seed1141091339",
    "no_trainw": ABLATION_ROOT / "no_trainw" / "seed1141091339",
}


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_comparison(report_dir, previous, completed):
    # Keep actual N=1000 and N=150000 evaluations as separate rows. Replacing
    # N in a formula while holding an old sampled q fixed would be a projection,
    # not a newly sampled result, and is deliberately not reported as actual data.
    rows = []
    for result in previous + completed:
        rows.append({
            "Experiment": result["Experiment"],
            "MC samples": result["number of SNN evaluation samples"],
            "Sampled SNN train error": result["SNN train error"],
            "SNN train error upper bound": result["SNN train error upper bound"],
            "SNN test error": result["SNN test error"],
            "Test error type": "sampled mean",
            "KL(Q || P)": result["KL(Q || P)"],
            "PAC-Bayes bound": result["PAC-Bayes bound"],
            "confidence": result["confidence"],
            "Source": "saved posterior evaluation",
        })
    # Paper SNN errors in Table 1 are corrected upper bounds, whereas our
    # "SNN test error" is an uncorrected sampled mean. Label that distinction
    # rather than treating the two quantities as directly identical metrics.
    rows.append({
        "Experiment": "Paper T600",
        "MC samples": 150000,
        "Sampled SNN train error": None,
        "SNN train error upper bound": 0.028,
        "SNN test error": 0.034,
        "Test error type": "paper reported upper bound",
        "KL(Q || P)": 5144,
        "PAC-Bayes bound": 0.161,
        "confidence": 0.965,
        "Source": "https://arxiv.org/html/1703.11008 (Table 1; Sections 3.3 and 4.4)",
    })
    write_json(report_dir / "comparison.json", rows)
    with (report_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Monte Carlo evaluation comparison", "",
        "The saved posteriors are evaluated without retraining. Existing results are preserved.",
        "Paper values are from Table 1 and Sections 3.3/4.4 of https://arxiv.org/html/1703.11008.",
        "The paper reports SNN train/test error upper bounds; our test errors are sampled means.",
        "Every confidence value is per experiment, not a joint guarantee for all rows.", "",
        "| Experiment | Samples | Sampled train error | Train error upper | PAC bound |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        raw = row["Sampled SNN train error"]
        raw_text = "not reported" if raw is None else f"{100 * raw:.4f}%"
        lines.append(f"| {row['Experiment']} | {row['MC samples']} | {raw_text} | "
                     f"{100 * row['SNN train error upper bound']:.4f}% | "
                     f"{100 * row['PAC-Bayes bound']:.4f}% |")
    (report_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snn_samples", type=int, default=150000)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", nargs="+", choices=list(POSTERIORS), default=list(POSTERIORS))
    args = parser.parse_args()
    if args.snn_samples < 1:
        parser.error("--snn_samples must be positive")
    report_dir = (args.output_dir or RESULTS / (
        f"evaluation_mc{args.snn_samples}_{datetime.now():%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    )).resolve()
    report_dir.mkdir(parents=True, exist_ok=False)
    previous = [row for row in read_json(PREVIOUS_COMPARISON) if row["Experiment"] in args.conditions]
    protected = {PREVIOUS_COMPARISON}
    for name in args.conditions:
        source = POSTERIORS[name]
        models = list(source.glob("model_mean_*.pickle"))
        if len(models) != 1:
            raise ValueError(f"Expected one complete posterior model in {source}")
        protected.update(models)
        protected.add(source / "final_summary.json")
    # Hash trained models and existing summaries before and after the suite to
    # verify that reevaluation preserves the original experimental artifacts.
    before = {str(path): file_hash(path) for path in sorted(protected)}
    write_json(report_dir / "protected_hashes_before.json", before)
    status = {
        "state": "running", "samples_per_experiment": args.snn_samples,
        "conditions": args.conditions, "completed": [], "report_dir": str(report_dir),
        "started_at": datetime.now().isoformat(),
    }
    write_json(report_dir / "status.json", status)
    completed = []
    export_comparison(report_dir, previous, completed)
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    print(f"Report directory: {report_dir}", flush=True)
    try:
        for name in args.conditions:
            # Evaluate sequentially on the shared GPU to avoid four competing
            # long MC jobs. This is separate from the concurrent training runner.
            stem = "baseline" if name == "T600 baseline" else name
            target = report_dir / f"{stem}_summary.json"
            command = [sys.executable, "-u", str(PKG_ROOT / "experiments" / "recompute_summary.py"),
                       "--output_dir", str(POSTERIORS[name]), "--snn_samples", str(args.snn_samples),
                       "--summary_path", str(target), "--device", args.device]
            start = time.monotonic()
            status.update(current_experiment=name, current_samples=0, stage_eta_seconds=None)
            write_json(report_dir / "status.json", status)
            print(f"Starting {name}: {args.snn_samples} posterior draws", flush=True)
            with (report_dir / f"{stem}_evaluation.log").open("x", encoding="utf-8") as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace", env=environment) as process:
                    status["evaluation_pid"] = process.pid
                    write_json(report_dir / "status.json", status)
                    first_progress_time = None
                    first_progress_count = None
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        match = re.search(r"Final posterior evaluation: (\d+)/(\d+)", line)
                        if match:
                            count, total = map(int, match.groups())
                            now = time.monotonic()
                            if first_progress_time is None:
                                first_progress_time, first_progress_count = now, count
                            rate = ((count - first_progress_count) / (now - first_progress_time)
                                    if count > first_progress_count else None)
                            status.update(current_samples=count, elapsed_stage_seconds=now - start,
                                          stage_eta_seconds=(total - count) / rate if rate else None,
                                          updated_at=datetime.now().isoformat())
                            write_json(report_dir / "status.json", status)
                            if count % 5000 == 0 or count == total or count == 200:
                                eta = status["stage_eta_seconds"]
                                eta_text = f"{eta / 3600:.2f} hours" if eta is not None else "calculating"
                                print(f"{name}: {count}/{total}; stage ETA: {eta_text}", flush=True)
                        elif "evaluation started" in line:
                            print(line.strip(), flush=True)
                    return_code = process.wait()
            if return_code:
                raise RuntimeError(f"{name} failed with exit code {return_code}; see {stem}_evaluation.log")
            result = read_json(target)
            result["Experiment"] = name
            completed.append(result)
            status["completed"].append(name)
            export_comparison(report_dir, previous, completed)
            write_json(report_dir / "status.json", status)
            print(f"Completed {name}: PAC bound = {result['PAC-Bayes bound']:.8f}", flush=True)
        after = {path: file_hash(Path(path)) for path in before}
        write_json(report_dir / "protected_hashes_after.json", after)
        if before != after:
            raise RuntimeError("A protected input changed during evaluation; inspect the hash manifests")
        status.update(state="complete", finished_at=datetime.now().isoformat(), protected_inputs_unchanged=True)
        write_json(report_dir / "status.json", status)
        print(f"All evaluations complete: {report_dir / 'comparison.md'}", flush=True)
    except BaseException as error:
        status.update(state="failed", error=str(error), updated_at=datetime.now().isoformat())
        write_json(report_dir / "status.json", status)
        raise


if __name__ == "__main__":
    main()
