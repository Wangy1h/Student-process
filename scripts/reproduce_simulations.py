"""Explicit, resumable generation of the experiments used in this paper.

By default this program only shows help.  --list also makes no random draws.
Use --block NAME --output DIR or --all --output DIR to generate new outputs.
The saved manuscript archive is never overwritten.  Once all 27 blocks are
complete, their 98 final arrays are assembled as simulation_arrays.npz in DIR.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import scipy
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data" / "config.json"
sys.dont_write_bytecode = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an unverified output: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def residual_mle(q: np.ndarray, m: int, nu: float, *, root_iterations: int = 64,
                 root_score_tolerance: float = 1e-10) -> tuple[np.ndarray, float]:
    """Solve the residual score for each row, using the original log bisection.

    Columns are independent trajectories; rows are Monte Carlo datasets.
    This is the residual-likelihood estimator, not the full-vector joint MLE.
    The returned error is the maximum absolute *normalized* score residual.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not q.shape[0] or not q.shape[1]:
        raise ValueError("q must be a nonempty two-dimensional residual array")
    if not np.isfinite(q).all() or not (q > 0).all() or m <= 0 or nu <= 0:
        raise ValueError("Residuals, m and nu must be positive and finite")
    logq = np.log(q)
    lo = np.min(logq, axis=1) - np.log(m)
    hi = np.max(logq, axis=1) - np.log(m)
    target = m / (m + nu)
    for _ in range(root_iterations):
        mid = (lo + hi) / 2
        score = special.expit(logq - np.log(nu) - mid[:, None]).mean(1) - target
        lo = np.where(score > 0, mid, lo)
        hi = np.where(score > 0, hi, mid)
    estimate = np.exp((lo + hi) / 2)
    error = float(np.max(abs(special.expit(
        logq - np.log(nu) - np.log(estimate)[:, None]).mean(1) - target)))
    if not np.isfinite(estimate).all() or not (estimate > 0).all():
        raise ArithmeticError("The residual MLE is not positive and finite")
    if error >= root_score_tolerance:
        raise ArithmeticError(f"Normalized score residual {error} exceeds tolerance")
    return estimate, error


def brownian(cfg: dict, design: str) -> tuple[str, dict, dict]:
    c = cfg["brownian"]
    rng = np.random.default_rng(c["seed"] + (design == "fixed"))
    b, finest_n = c["replications"], max(c["sizes"])
    # Draw all normals first, then one mixing vector for each nu: this order
    # reproduces the original experiment and its cross-grid dependence.
    z = rng.standard_normal((b, finest_n))
    v = np.stack([nu / rng.chisquare(nu, b) for nu in c["nu"]])
    arrays = {}
    maximum_endpoint_error = 0.0
    for k, nu in enumerate(c["nu"]):
        w = cfg["theta"] * v[k]
        previous_mu = None
        for n in c["sizes"]:
            if design == "fixed":
                if finest_n % n:
                    raise ValueError("Fixed-interval grids must divide the finest grid")
                dt = c["fixed_T"] / n
                dg = z.reshape(b, n, finest_n // n).sum(2) * np.sqrt(c["fixed_T"] / finest_n)
            else:
                dt = c["delta"]
                dg = z[:, :n] * np.sqrt(dt)
            dx = cfg["mu"] * dt + np.sqrt(w)[:, None] * dg
            muhat = dx.sum(1) / (n * dt)
            q = np.sum((dx - muhat[:, None] * dt) ** 2, axis=1) / dt
            if design == "fixed" and previous_mu is not None:
                maximum_endpoint_error = max(maximum_endpoint_error,
                    float(np.max(abs(muhat - previous_mu))))
            previous_mu = muhat
            arrays[f"nu{nu}_n{n}"] = np.stack([muhat, q, w])
    if maximum_endpoint_error >= 1e-11:
        raise ArithmeticError("The shared fixed-interval endpoint changed across grids")
    return f"brownian_{design}_statistics.npz", arrays, {
        "valid_datasets": b, "seed": c["seed"] + (design == "fixed"),
        "max_fixed_endpoint_error": maximum_endpoint_error,
    }


def coverage(cfg: dict) -> tuple[str, dict, dict]:
    c = cfg["coverage"]
    rng = np.random.default_rng(c["seed"])
    b = c["replications"]
    arrays = {}
    # W is shared across n within each nu, while A and U are redrawn at each n.
    for nu in c["nu"]:
        w = cfg["theta"] * nu / rng.chisquare(nu, b)
        for n in c["sizes"]:
            a = rng.standard_normal(b)
            u = rng.chisquare(n - 1, b)
            arrays[f"nu{nu}_n{n}"] = np.stack([w, a, u])
    return "coverage_raw.npz", arrays, {"valid_datasets": b, "seed": c["seed"]}


def replicated(cfg: dict, m: int, nu: float, design_n: int | None = None) -> tuple[str, dict, dict]:
    if design_n is None:
        c = cfg["replicated"]
        js = c["J"]
        seed = c["seed"] + 100 * m + nu
        key = f"replicated_m{m}_nu{nu}"
    else:
        c = cfg["design"]
        js = [c["budget"] // design_n]
        seed = c["seed"] + design_n
        key = f"design_n{design_n}"
    rng = np.random.default_rng(seed)
    b, jmax = c["replications"], max(js)
    # The denominator is drawn before the numerator, as in the original code.
    q = cfg["theta"] * nu / rng.chisquare(nu, (b, jmax)) * rng.chisquare(m, (b, jmax))
    arrays = {}
    score_errors = {}
    for j in js:
        estimate, error = residual_mle(q[:, :j], m, nu,
            root_iterations=cfg["numerics"]["root_iterations"],
            root_score_tolerance=cfg["numerics"]["root_score_tolerance"])
        arrays[f"J{j}_residual_likelihood"] = estimate
        score_errors[str(j)] = error
    return key + "_estimates.npz", arrays, {
        "valid_datasets": b, "seed": seed, "m": m, "nu": nu,
        "trajectory_counts": js, "root_max_score": score_errors,
    }


def blocks(cfg: dict) -> dict:
    jobs = {"brownian_long": lambda: brownian(cfg, "long"),
            "brownian_fixed": lambda: brownian(cfg, "fixed"),
            "coverage": lambda: coverage(cfg)}
    jobs.update({f"replicated_m{m}_nu{nu}":
        lambda m=m, nu=nu: replicated(cfg, m, nu)
        for m in cfg["replicated"]["m"] for nu in cfg["replicated"]["nu"]})
    jobs.update({f"design_n{n}":
        lambda n=n: replicated(cfg, n - 1, cfg["design"]["nu"], n)
        for n in cfg["design"]["sizes"]})
    return jobs


def validate_output_directory(output: Path) -> None:
    # Generated archives must be kept outside the manuscript delivery directory
    # and the historical source projects. This also protects saved data archives.
    protected = [ROOT.resolve()]
    protected.extend(p.resolve() for p in ROOT.parent.glob("revision-*") if p.is_dir())
    protected.append((ROOT.parent / "parameter_estimate_SJS_revision").resolve())
    for project in protected:
        if output == project or project in output.parents:
            raise ValueError(f"Choose a new output directory outside the manuscript projects: {output}")


def verify_complete(output: Path, entry: dict, config_hash: str, script_hash: str) -> bool:
    if entry.get("status") != "complete":
        return False
    if entry.get("config_sha256") != config_hash or entry.get("script_sha256") != script_hash:
        raise RuntimeError("A completed block belongs to different config/code; use a new output directory")
    for filename, expected in entry["outputs"].items():
        path = output / filename
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Completed output is missing or changed: {path}; inspect before rerunning")
    return True


def assemble_archive(output: Path, state: dict, jobs: dict, config_hash: str, script_hash: str) -> None:
    if not all(key in state["blocks"] and verify_complete(output, state["blocks"][key],
            config_hash, script_hash) for key in jobs):
        return
    path = output / "simulation_arrays.npz"
    existing = state.get("archive", {})
    if existing.get("sha256"):
        if path.exists() and sha256(path) == existing["sha256"]:
            return
        raise RuntimeError("The assembled archive changed or is missing; inspect before rerunning")
    arrays = {}
    for key in jobs:
        filename = state["blocks"][key]["array_file"]
        with np.load(output / filename, allow_pickle=False) as saved:
            arrays.update({Path(filename).stem + "/" + name: saved[name] for name in saved.files})
    save_arrays(path, arrays)
    state["archive"] = {"filename": path.name, "sha256": sha256(path), "array_count": len(arrays)}
    save_json(output / "manifest.json", state)
    print(f"ASSEMBLED {path}: {len(arrays)} arrays", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="List blocks; do not draw random numbers or write files")
    action.add_argument("--block", metavar="NAME", help="Generate one block, or skip it after hash verification")
    action.add_argument("--all", action="store_true", help="Explicitly generate every block")
    parser.add_argument("--output", type=Path, help="New output directory, outside all manuscript projects")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Configuration (default: data/config.json)")
    args = parser.parse_args(argv)
    if not (args.list or args.block or args.all):
        parser.print_help()
        return 0
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = blocks(cfg)
    if args.list:
        print("\n".join(jobs))
        return 0
    if args.block and args.block not in jobs:
        parser.error(f"Unknown block {args.block}; use --list")
    if args.output is None:
        parser.error("--output is required for generation")
    output = args.output.resolve()
    validate_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    script_hash = sha256(Path(__file__))
    manifest = output / "manifest.json"
    state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {
        "schema_version": 1, "config_sha256": config_hash, "script_sha256": script_hash,
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform(), "generator": "NumPy default_rng PCG64"},
        "blocks": {},
    }
    if state.get("config_sha256") != config_hash or state.get("script_sha256") != script_hash:
        raise RuntimeError("Output manifest belongs to different config/code; use a new output directory")
    snapshot = output / "config.json"
    if snapshot.exists() and sha256(snapshot) != config_hash:
        raise RuntimeError("Output config snapshot changed; inspect before rerunning")
    if not snapshot.exists():
        snapshot.write_bytes(config_path.read_bytes())
    save_json(manifest, state)
    selected = list(jobs) if args.all else [args.block]
    for key in selected:
        old = state["blocks"].get(key, {})
        if verify_complete(output, old, config_hash, script_hash):
            print(f"SKIP verified completed block: {key}", flush=True)
            continue
        attempt = {"status": "running", "started_utc": utc_now(),
                   "config_sha256": config_hash, "script_sha256": script_hash,
                   "attempt": old.get("attempt", 0) + 1}
        if old:
            attempt["previous_attempts"] = old.get("previous_attempts", []) + [
                {name: value for name, value in old.items() if name != "previous_attempts"}]
        state["blocks"][key] = attempt
        save_json(manifest, state)
        print(f"RUN {key}", flush=True)
        start = time.perf_counter()
        try:
            filename, arrays, diagnostics = jobs[key]()
            if not all(np.isfinite(array).all() for array in arrays.values()):
                raise ArithmeticError("Generated arrays contain nonfinite values")
            path = output / filename
            save_arrays(path, arrays)
            attempt.update(status="complete", finished_utc=utc_now(),
                elapsed_seconds=time.perf_counter() - start, array_file=filename,
                outputs={filename: sha256(path)}, diagnostics=diagnostics,
                arrays={name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                        for name, array in arrays.items()})
        except Exception:
            attempt.update(status="failed", finished_utc=utc_now(),
                elapsed_seconds=time.perf_counter() - start, error=traceback.format_exc())
            save_json(manifest, state)
            raise
        save_json(manifest, state)
        print(f"DONE {key} ({attempt['elapsed_seconds']:.2f} s)", flush=True)
    assemble_archive(output, state, jobs, config_hash, script_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
