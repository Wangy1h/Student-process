# Drift and scale inference for Student-t processes

Python code and saved simulation outputs for the manuscript of this title.
The model is

```text
X_t = mu*t + sqrt(theta*V)*G_t,  W = theta*V,
V = nu/Z,  Z ~ chi-square(nu).
```

Here `theta` is a fixed scale parameter, `V` is the mixing variable, and `W` is
the realized scale. One draw of `V` is shared by all observations of a trajectory;
independent trajectories have independent mixing variables and Gaussian processes.
The simulation examples use the Brownian Gaussian kernel.

## Contents

- `scripts/plot_figures.py`: rebuild four vector PDF figures and numerical summaries
  from the saved arrays, with no random draws or refitting.
- `scripts/reproduce_simulations.py`: explicitly regenerate the original simulation
  blocks, with seeds, settings, solver checks, and resumable manifests.
- `scripts/verify_archive.py`: compare regenerated arrays with the saved outputs.
- `data/config.json`: parameters, random seeds, sample sizes, and solver settings.
- `data/simulation_arrays.npz`: 98 saved numerical arrays (synthetic data only).
- `figures/`: `recovery.pdf`, `random_limits.pdf`, `replication.pdf`, and
  `replication_normal.pdf`.
- `results/`: six CSV summaries computed from the saved arrays.

The repository contains no manuscript, referee reports, third-party articles,
private environment files, or participant data. All file paths used for reproduction
are relative to this repository or supplied on the command line.

## Environment and quick reproduction

Validated environment: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, and
Matplotlib 3.11.2. The dependency versions are pinned in `requirements.txt`.
From the repository root, run:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/plot_figures.py --summaries-dir results
```

This regenerates the four figures and six CSV files without performing simulations.
It also exports a LaTeX fragment of the complete replicated-trajectory table; that
generated fragment is not tracked in the repository. An optional `--audit-dir audit`
saves alignment checks, hashes, and the running environment.

Manuscript table checks are disabled by default because the manuscript is not
included. Authors with the private source can additionally pass
`--manuscript-dir PATH` to compare the numerical rows of four tables. This option
does not check narrative text and never changes the manuscript.

The bundled data archive has SHA-256:

```text
9ec146f3e14b30b6635af9455d9ef2304ed918dadf66636f753f213eac49801a
```

## Optional regeneration from the original seeds

Expensive simulation regeneration is opt-in. Listing blocks does not draw random
numbers or write simulation outputs:

```bash
python scripts/reproduce_simulations.py --list
python scripts/reproduce_simulations.py --block replicated_m7_nu10 --output ../student_t_regenerated
# Run only when all simulations need to be regenerated:
python scripts/reproduce_simulations.py --all --output ../student_t_regenerated
python scripts/verify_archive.py ../student_t_regenerated/simulation_arrays.npz
```

Use a new output directory outside this repository. Repeating an identical command
verifies hashes and skips completed blocks. Changed code/configuration or damaged
completed outputs require investigation or a new directory; the program will not
silently overwrite them. A manifest records environment, hashes, block status,
failures, dimensions, and normalized score errors. After all 27 blocks finish,
the script assembles the 98 arrays into a new archive.

The 27 blocks comprise two Brownian designs, one exact-interval experiment,
15 replicated-trajectory settings, and nine allocation settings. The generator
retains the original PCG64 seeds and random-call order. Within each replicated
setting, smaller `J` uses a prefix of the same residual array. The residual likelihood
uses 64 log-scale bisection iterations and a normalized score tolerance of `1e-10`.
The residual estimator is distinct from the full-vector joint MLE.

Archive bytes can differ because ZIP metadata differs. `verify_archive.py` checks
array names, dimensions, dtypes, and values instead of comparing archive bytes.
Exact numerical reproduction should be assessed in the pinned environment.

## Reading the outputs

`n` counts observations within a trajectory, `m=n-1` is the residual dimension,
`a_n` controls drift precision, `J` counts independent trajectories per dataset,
and `M` counts Monte Carlo datasets. The Brownian designs use `M=6000`, exact
intervals use `M=12000`, and replicated/allocation experiments use `M=5000`.
The configuration specifies every setting and seed.

The saved arrays are grouped by experiment-name prefixes; use
`numpy.load('data/simulation_arrays.npz', allow_pickle=False).files` to inspect them.
Brownian blocks contain drift estimates, residual sums of squares, and realized
scales. Coverage blocks contain realized scales and Gaussian/chi-square pivot
draws. Replicated/allocation blocks contain residual-likelihood estimates.
CSV summaries retain setting labels and valid Monte Carlo sample counts.

In `recovery.pdf`, error bars are 1.96 estimated Monte Carlo standard errors.
In `random_limits.pdf`, boxes show quartiles and whiskers show 5th/95th percentiles.
In `replication.pdf`, error bars describe Monte Carlo uncertainty; dashed error
curves are asymptotic benchmarks and the horizontal coverage line is 0.95.
`replication_normal.pdf` compares standardized estimation errors with normal
quantiles. These are estimator-specific simulations, not proofs of impossibility
for all estimators. Raw bias/RMSE summaries are restricted as in the original
analysis; heavy-tail squared-loss means can have unstable Monte Carlo behavior.

## Verification and scope

For repository preparation, the saved archive was unchanged, all four figures
were regenerated without random draws, and their bytes matched the manuscript
figures in the pinned environment. The six CSV summaries matched the previously
validated outputs. The generation script was preserved byte for byte. This
packaging step did not rerun the 27 original-scale simulation blocks or alter the
model, estimators, data, or reported statistics.

Plotting changes separate optional private manuscript table checks from standalone
figure reproduction and replace the external layout helper with a self-contained
check for these equal-size row panels. Figure construction and statistical
calculations are unchanged. The manuscript and its references are not changed.
No reuse licence is assigned in this initial code package; a licence can be added
by the copyright holders. For citation, retain the exact repository commit used;
no release DOI or permanent archival identifier has been assigned here.
