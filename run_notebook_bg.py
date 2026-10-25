"""Run the code cells of a notebook as a plain Python script (no Jupyter needed).

Intended for long runs in the background, e.g. N-fold calibration with many folds.
The notebook itself is not modified: a snapshot of it is copied to the output folder.

- Figures are saved as PNG (cellNN_figM.png) instead of being shown.
- Printed output of all cells goes to the console (redirect it to a log file).
- n_folds_results / calibrated_parameters are saved as CSV / JSON after the n-fold cell.

Usage (from the watmod_exercises folder):
    python run_notebook_bg.py gsteig_run.ipynb --folds 50 --objective kge --bound k=0.1,3 > run.log 2>&1
"""

import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no windows: figures are saved to files
import matplotlib.pyplot as plt


def set_folds(source: str, folds: int) -> str:
    """Replace the folds=... argument in the n-fold set_options cell."""
    marker = "folds=5"
    if marker not in source:
        raise ValueError(f"'{marker}' not found in the n-fold cell")
    return source.replace(marker, f"folds={folds}")


def set_bound(source: str, name: str, low: float, high: float) -> str:
    """Replace the bounds of one parameter, e.g. "k": (1, 3) -> "k": (0.1, 3)."""
    pattern = rf'"{name}": \([^)]*\)'
    if not re.search(pattern, source):
        raise ValueError(f"bounds for '{name}' not found in the n-fold cell")
    return re.sub(pattern, f'"{name}": ({low}, {high})', source)


def set_objective(source: str, objective: str) -> str:
    """Replace objective_function="..." in a set_options cell."""
    return re.sub(r'objective_function="\w+"', f'objective_function="{objective}"', source)


def run_cell(source: str, namespace: dict) -> None:
    """Execute a cell; print the value of a trailing expression like Jupyter does."""
    tree = ast.parse(source)
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), namespace)
    if last_expr is not None:
        value = eval(compile(last_expr, "<cell>", "eval"), namespace)
        if value is not None:
            print(value)


def save_figures(out_dir: Path, cell_index: int) -> None:
    for n, num in enumerate(plt.get_fignums(), start=1):
        plt.figure(num).savefig(out_dir / f"cell{cell_index:02d}_fig{n}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook")
    parser.add_argument("--folds", type=int, default=50, help="number of folds for the n-fold calibration")
    parser.add_argument("--objective", default=None,
                        help="objective function for all calibrations, e.g. nse, kge (default: as in the notebook)")
    parser.add_argument("--bound", action="append", default=[], metavar="NAME=LOW,HIGH",
                        help="override the bounds of a parameter in the n-fold cell, e.g. --bound k=0.1,3")
    parser.add_argument("--out", default=None, help="output folder (default: runs/<notebook>_<objective>_nfolds<N>_<time>)")
    parser.add_argument("--stop-after", type=int, default=None, help="stop after this cell index (for testing)")
    args = parser.parse_args()

    nb_path = Path(args.notebook).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = f"{args.objective}_" if args.objective else ""
    out_dir = Path(args.out) if args.out else nb_path.parent / "runs" / f"{nb_path.stem}_{tag}nfolds{args.folds}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nb_path, out_dir / nb_path.name)  # snapshot of the notebook that was run

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    nfold_cells = [i for i, s in code_cells if 'method = "n-folds"' in s]
    if len(nfold_cells) != 1:
        raise ValueError(f"expected one n-fold set_options cell, found {nfold_cells}")

    # Relative paths in the notebook (data/..., images/...) and imports refer to the notebook folder
    os.chdir(nb_path.parent)
    sys.path.insert(0, str(nb_path.parent))

    print(f"notebook:  {nb_path}\noutput:    {out_dir}\nfolds:     {args.folds}\n"
          f"objective: {args.objective or 'as in the notebook'}\n", flush=True)
    namespace = {"__name__": "__main__"}
    t_start = time.time()

    for i, source in code_cells:
        if args.stop_after is not None and i > args.stop_after:
            break
        if not source.strip():
            continue
        if i == nfold_cells[0]:
            source = set_folds(source, args.folds)
            for b in args.bound:
                name, values = b.split("=")
                low, high = (float(v) for v in values.split(","))
                source = set_bound(source, name, low, high)
                print(f"(cell {i}: bounds of {name} set to ({low}, {high}))", flush=True)
        if args.objective and "objective_function=" in source:
            source = set_objective(source, args.objective)
            print(f"(cell {i}: objective_function set to {args.objective})", flush=True)

        print(f"\n===== cell {i} ({datetime.now():%H:%M:%S}, {(time.time() - t_start) / 60:.1f} min) =====", flush=True)
        t_cell = time.time()
        try:
            run_cell(source, namespace)
        except Exception:
            traceback.print_exc()
            print(f"cell {i} failed: stopping here", flush=True)
            save_figures(out_dir, i)
            sys.exit(1)
        save_figures(out_dir, i)
        print(f"----- cell {i} done in {time.time() - t_cell:.1f} s", flush=True)

        # Save the n-fold results as soon as they exist
        if "n_folds_results" in namespace and not (out_dir / "n_folds_results.csv").exists():
            namespace["n_folds_results"].to_csv(out_dir / "n_folds_results.csv", index=False)
            params = {k: float(v) for k, v in namespace["calibrated_parameters"].items()}
            (out_dir / "calibrated_parameters.json").write_text(json.dumps(params, indent=2))
            print(f"saved n_folds_results.csv and calibrated_parameters.json", flush=True)

    print(f"\nfinished in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
