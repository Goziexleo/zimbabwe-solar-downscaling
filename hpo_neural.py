"""Honest hyperparameter search for the CNN and U-Net (Section 3.6.7).

Table 3.2's neural hyperparameters were chosen by a grid search scored on
"validation MSE" - the withheld 2011-2024 record the models are then reported
against. That is selection on the evaluation set, the same defect corrected for
XGBoost's boosting rounds (Section 6.12) and for both networks' epoch counts
(Section 6.13), and it was the one instance left live.

This runs the same grids entirely inside the training period. Each candidate is
fitted on 1985-2004 and scored on an inner selection split of 2005-2010, exactly
as Phase A of the deployed training procedure does. Nothing in this search sees
2011-2024, so the winner is chosen without reference to the record it will be
judged on.

The grids are Table 3.2's, unchanged, so the comparison is like for like:

    CNN    learning rate  0.0005 / 0.001     loss weighting  0.001 / 0.01
    U-Net  learning rate  0.0002 / 0.0005 / 0.001

Seven fits. Each runs the real training script with PHASE_A_ONLY=1, so the
search uses the deployed model code rather than a reimplementation - the U-Net
optimisation sweep found that a reimplemented baseline does not reproduce the
deployed model's behaviour, and that lesson applies here.

    python hpo_neural.py                 # run the search
    python hpo_neural.py --dry-run       # print the grid and exit

Writes data/processed/evaluation/neural_hpo.csv.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data/processed/evaluation/neural_hpo.csv")

# Table 3.2, verbatim.
CNN_GRID = {"lr": [0.0005, 0.001], "lambda_gp": [0.001, 0.01]}
UNET_GRID = {"lr": [0.0002, 0.0005, 0.001]}

# What is deployed today, for the comparison the write-up needs.
DEPLOYED = {"CNN": {"lr": 0.0005, "lambda_gp": 0.01},
            "U-Net": {"lr": 0.0002}}


def candidates():
    for lr, lam in itertools.product(CNN_GRID["lr"], CNN_GRID["lambda_gp"]):
        yield ("CNN", {"lr": lr, "lambda_gp": lam})
    for lr in UNET_GRID["lr"]:
        yield ("U-Net", {"lr": lr})


def run_one(model, cfg, log_dir):
    """Fit one candidate through Phase A and return its inner-selection MSE."""
    script = ("train_cnn_downscaler.py" if model == "CNN"
              else "train_unet_downscaler.py")
    env = dict(os.environ)
    env["PHASE_A_ONLY"] = "1"
    if model == "CNN":
        env["CNN_LEARNING_RATE"] = str(cfg["lr"])
        env["CNN_LAMBDA_GP"] = str(cfg["lambda_gp"])
    else:
        env["UNET_LEARNING_RATE"] = str(cfg["lr"])

    fd, res_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    env["PHASE_A_OUT"] = res_path

    tag = "%s_%s" % (model.replace("-", ""),
                     "_".join("%s%g" % (k, v) for k, v in sorted(cfg.items())))
    log_path = os.path.join(log_dir, tag + ".log")

    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([sys.executable, os.path.join(ROOT, script)],
                              cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    minutes = (time.time() - t0) / 60.0

    if proc.returncode != 0:
        # A candidate that cannot be fitted is a result, not a crash: record it
        # and carry on, rather than losing the candidates already run.
        print("    FAILED (exit %d) - see %s" % (proc.returncode, log_path))
        os.unlink(res_path)
        return None, None, minutes, log_path

    with open(res_path) as fh:
        r = json.load(fh)
    os.unlink(res_path)
    return r["inner_select_mse"], r["selected_epoch"], minutes, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid = list(candidates())
    print("Honest neural hyperparameter search (Section 3.6.7)")
    print("Fit 1985-2004, select on 2005-2010. The withheld record is not touched.\n")
    for model, cfg in grid:
        mark = " <- deployed" if cfg == DEPLOYED[model] else ""
        print("  %-6s %s%s" % (model, cfg, mark))
    print("\n%d candidates" % len(grid))
    if args.dry_run:
        return 0

    log_dir = os.path.join(ROOT, "logs", "hpo_neural")
    os.makedirs(log_dir, exist_ok=True)

    rows = []
    for i, (model, cfg) in enumerate(grid, 1):
        print("\n[%d/%d] %s %s" % (i, len(grid), model, cfg))
        mse, epoch, minutes, log_path = run_one(model, cfg, log_dir)
        if mse is None:
            print("    no result")
        else:
            print("    inner-select MSE %.6f at epoch %d (%.1f min)"
                  % (mse, epoch, minutes))
        rows.append(dict(model=model, **cfg, inner_select_mse=mse,
                         selected_epoch=epoch, minutes=round(minutes, 2),
                         is_deployed=(cfg == DEPLOYED[model]),
                         log=os.path.relpath(log_path, ROOT)))
        # Written after every candidate so an interrupted search keeps its work.
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    for model in ("CNN", "U-Net"):
        sub = df[(df.model == model) & df.inner_select_mse.notna()]
        if sub.empty:
            print("%s: no candidate completed" % model)
            continue
        best = sub.loc[sub.inner_select_mse.idxmin()]
        dep = sub[sub.is_deployed]
        print("\n%s" % model)
        print(sub.drop(columns=["log", "is_deployed"]).to_string(index=False))
        print("  best under honest selection: %s"
              % {k: best[k] for k in ("lr", "lambda_gp") if k in best and pd.notna(best[k])})
        if not dep.empty:
            d = dep.iloc[0]
            same = best.name == d.name
            print("  deployed configuration is %s"
                  % ("the same - Table 3.2 survives the correction" if same else
                     "DIFFERENT: %.6f deployed against %.6f best"
                     % (d.inner_select_mse, best.inner_select_mse)))

    print("\nSaved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
