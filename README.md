<p align="left">
  <img src="assets/comm_logo.png" alt="CoMM logo" width="128" style="vertical-align:middle; margin-right:10px;" />
  <span style="font-size: 22px; font-weight: 600;">CoMM</span>
  <span> — contraction-only MM for nonnegative CP/Tucker under β-divergences</span>
</p>

## Overview

**CoMM** is an efficient algoritmic framework for constrained nonnegative tensor factorization
under the entry-wise β-divergence family ($$0 \le \beta < 2$$), with a focus on:

- **Nonnegative CP decomposition**
- **Nonnegative Tucker decomposition**

### What “contraction-only” means ? 

Classical multiplicative updates for CP/Tucker are often presented via **mode unfoldings** (matricizations) and
large intermediate objects (e.g., explicit Khatri–Rao/Kronecker products, unfolded tensors).
CoMM instead expresses all update quantities **directly as tensor contractions** over index sets, implemented with
`einsum` backends (`numpy.einsum`, `opt_einsum`), so that the core algorithms never require explicit unfoldings.
If you’re new to `einsum`, a very accessible tutorial is: [“Einsum is All You Need”](https://rockt.ai/2018/04/30/einsum).

### Implemented algorithms

CoMM provides two MM variants:

- **B-CoMM**: **block-MM** (classical multiplicative updates) written in contraction form.
- **J-CoMM**: **joint-MM**: build a **single joint surrogate at a reference iterate**, precompute reference-powered
  tensors once, then perform a few **cheap inner updates** that reuse cached reference quantities.

In practice, J-CoMM reduces the cost of repeatedly forming powered tensors (e.g., $\widehat X^{\beta-1}$ and
$X \odot \widehat X^{\beta-2}$) by amortizing them across several inner steps.

### β-divergence family

We optimize the entry-wise β-divergence objective
$D_\beta(\mathcal X,\widehat{\mathcal X}) = \sum_i d_\beta(X_i \mid \widehat X_i),$
including common special cases such as:

- $\beta = 1$: generalized KL divergence
- $\beta = 0$: Itakura–Saito divergence

All factors (and the Tucker core) are maintained strictly positive via a small safeguard $\varepsilon>0$
to avoid numerical issues when evaluating powers like $\widehat X^{\beta-1}$.

## Core idea(s) 

We minimize an entry-wise β-divergence objective of the form
$D_\beta(\mathcal X,\widehat{\mathcal X}) = \sum_i d_\beta(X_i \mid \widehat X_i).$

### Block-MM (B-CoMM)
At each outer iteration, we construct a tight majorizing surrogate that is separable in the entries of one block
(one factor matrix, or the Tucker core), which yields a closed-form multiplicative update of the generic form
$\Theta_b \leftarrow \Theta_b \odot \left(\frac{\mathrm{Num}_b}{\mathrm{Den}_b}\right)^{\gamma(\beta)}.$
Here, $\mathrm{Num}_b$ and $\mathrm{Den}_b$ are obtained via contraction-only operators (einsum-style tensor
contractions), so no explicit unfolding is required.

### Joint-MM (J-CoMM)
J-CoMM constructs a single joint majorizer $G(\Theta\mid\widetilde\Theta)$ for all tensor parameters $\Theta$.
During an inner sweep, it updates one block at a time using closed-form multiplicative rules: when all other
blocks are fixed, the surrogate $G(\cdot\mid\widetilde\Theta)$ becomes entrywise separable in the active block,
so each block update is an exact minimizer of $G(\cdot\mid\widetilde\Theta)$ with respect to that block.

J-CoMM applies to both **nonnegative CP** and **nonnegative Tucker** models. The joint surrogate is built at a
reference iterate $\widetilde\Theta$ and then decreased by a few inexpensive inner sweeps. This structure also
enables efficient implementations that reuse reference-powered tensors computed once at $\widetilde\Theta$, e.g.
$\widetilde{\mathcal P} = X \odot \widetilde{\widehat X}^{\,\beta-2},
\qquad
\widetilde{\mathcal Q} = \widetilde{\widehat X}^{\,\beta-1}.$




## Quick start (recommended)

### Step 1: Install Python (3.9+)

If you don't have Python installed:

1. Go to [python.org](https://www.python.org/downloads/)
2. Download **Python 3.9+**
3. Install with default settings
4. Verify in a terminal:

```bash
python3 --version
```

If `python3` is not available on your system, try:

```bash
python --version
```

**Windows (PowerShell)**:

```powershell
py --version
python --version
```

### Step 2: Download CoMM

#### Option A (recommended): clone with git

```bash
git clone https://github.com/vleplat/CoMM.git
cd CoMM
```

**Windows (PowerShell)**:

```powershell
git clone https://github.com/vleplat/CoMM.git
cd CoMM
```

#### Option B: download ZIP

1. Go to the GitHub repository page
2. Click **Code** → **Download ZIP**
3. Extract it to a folder
4. Open a terminal in that folder

### Step 3: Create a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

**Windows (PowerShell)**:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

**Windows (cmd.exe)**:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
```

### Step 4: Install CoMM

```bash
python -m pip install -e .
```

**Windows (PowerShell)**:

```powershell
python -m pip install -e .
```

### Step 5 (optional): Install competitor dependencies

To enable NNEinFact comparisons (PyTorch):

```bash
python -m pip install -e ".[competitors]"
```

**Windows (PowerShell)**:

```powershell
python -m pip install -e ".[competitors]"
```

## Competitor setup (NNEinFact)

This repo integrates the reference `einfact.py` implementation from Hood & Schein.

- **Upstream repo**: [jhood3/einfact](https://github.com/jhood3/einfact)

By default, scripts expect a local copy at the **project root**:

```bash
ls -la einfact.py
```

Example (fetch a single file):

```bash
curl -L -o einfact.py https://raw.githubusercontent.com/jhood3/einfact/main/einfact.py
```

**Note:** our wrapper may patch `einfact.py` at runtime (small upstream fix + optionally disable their internal validation split). If you want to keep a pristine vendor copy, store it elsewhere and pass its path via `--einfact_path`.



## Running the scripts

All scripts are modules under `comm.scripts` and should be executed with `python -m ...`.
Figures are saved under `figures/` by default.

On Windows, run the same commands from **PowerShell** once your virtual environment is activated.

### 1) Synthetic CP benchmark

```bash
python -m comm.scripts.bench_cp \
  --beta 1.5 --shape 60 52 44 36 --rank 8 --iters 40 --inner 1 \
  --no_show --logy --start_iter 1
```

With competitor:

```bash
python -m comm.scripts.bench_cp \
  --beta 1.5 --shape 60 52 44 36 --rank 8 --iters 40 --inner 1 \
  --run_einfact --einfact_path ./einfact.py --device cpu --torch_threads 1 \
  --no_show --logy --start_iter 1
```

### 2) Synthetic Tucker benchmark

```bash
python -m comm.scripts.bench_tucker \
  --beta 1.5 --shape 60 52 44 36 --ranks 8 7 6 5 --iters 40 --inner 1 \
  --no_show --logy --start_iter 1
```

With competitor:

```bash
python -m comm.scripts.bench_tucker \
  --beta 1.5 --shape 60 52 44 36 --ranks 8 7 6 5 --iters 40 --inner 1 \
  --run_einfact --einfact_path ./einfact.py --device cpu --torch_threads 1 \
  --no_show --logy --start_iter 1
```

### 3) Real-data benchmark (Uber pickups tensor)

We use the *Uber pickups* dataset in the same tensor format as the NNEinFact demo: a nonnegative **5-way**
count tensor $X\in\mathbb{R}_+^{27\times 24\times 7\times 100\times 100}$ whose modes correspond to:

- week (27)
- hour (24)
- day-of-week (7)
- latitude index (100)
- longitude index (100)

We fit a **nonnegative Tucker** model with multilinear ranks $(10,10,5,10,10)$ under the **β-divergence** loss.
For this real-data experiment, we focus on optimization performance and report the **normalized (mean) objective**
value versus (i) outer iteration and (ii) wall-clock CPU time.

Run our methods (Tucker B-CoMM and J-CoMM):

```bash
python -m comm.scripts.bench_real_uber \
  --data data/Y.npz --key Y \
  --beta 1.5 --iters 40 --inner 1 \
  --tucker_ranks 10 10 5 10 10 \
  --no_show --logy --start_iter 1
```

#### Uber + NNEinFact (custom model from their demo/paper)

```bash
python -m comm.scripts.bench_real_uber \
  --data data/Y.npz --beta 1.5 --iters 40 \
  --tucker_ranks 10 10 5 10 10 \
  --run_einfact --einfact_path ./einfact.py \
  --einfact_model custom --k 6 --r 10 \
  --device cpu --torch_threads 1 \
  --plot_hour_factor \
  --no_show --logy --start_iter 1
```

#### Uber + NNEinFact (Tucker model for apples-to-apples *model class*)

This runs NNEinFact with the Tucker einsum string:
`wa,hb,dc,if,je,abcfe->whdij`
and ranks `(a,b,c,f,e) = (10,10,5,10,10)`.

```bash
python -m comm.scripts.bench_real_uber \
  --data data/Y.npz --beta 1.5 --iters 40 --inner 1 \
  --tucker_ranks 10 10 5 10 10 \
  --run_einfact --einfact_path ./einfact.py \
  --einfact_model tucker --einfact_tucker_ranks 10 10 5 10 10 \
  --device cpu --torch_threads 1 \
  --plot_hour_factor --plot_tucker_components --n_components 3 \
  --no_show --logy --start_iter 1
```

### 4) Apples-to-apples benchmark (thread control + time-to-target)

This script is designed to compare methods fairly by controlling threads and reporting **time-to-target loss**:

```bash
python -m comm.scripts.bench_apples \
  --problem tucker --beta 1.5 \
  --shape 60 52 44 36 --ranks 8 7 6 5 \
  --iters 40 --inner 1 --seed 0 \
  --threads 1 \
  --run_einfact --einfact_path ./einfact.py --torch_threads 1 \
  --device cpu
```

Notes:

- `--threads` sets common BLAS/Accelerate thread env vars (best-effort).
- `--torch_threads` sets `torch.set_num_threads(n)` for the competitor.

### 5) Seed/thread sweep (mean ± std curves)

Runs multiple seeds and overlays competitor curves for different Torch thread counts, saving a single figure:

```bash
python -m comm.scripts.bench_sweep \
  --problem tucker --beta 1.5 \
  --shape 80 70 60 50 --ranks 10 9 8 7 \
  --iters 40 --inner 1 \
  --n_seeds 5 --seed0 0 \
  --threads 1 \
  --run_einfact --einfact_path ./einfact.py --torch_threads_list 1 4 8 \
  --device cpu \
  --logy --start_iter 1
```

### 6) Tutorial: use CoMM as a Python library (direct API)

You do **not** need an extra script inside this repo to use CoMM on your own tensor.
Once you installed CoMM (`python -m pip install -e .`), you can either:

- **Option A**: write a small Python file (recommended for reproducibility), or
- **Option B**: run a one-off snippet directly from your terminal.

Below, `X` is your nonnegative NumPy tensor (dense `ndarray`).

#### CP decomposition (B-CoMM / J-CoMM)

```python
import numpy as np
from comm.models.cp import cp_bcomm, cp_jcomm

X = ...  # ndarray, shape (I1,...,IN), X >= 0
beta = 1.5
R = 10
eps = 1e-12
n_outer = 100
n_inner = 1

rng = np.random.default_rng(0)
A0 = [rng.random((X.shape[n], R)) + 1e-3 for n in range(X.ndim)]

A_b, loss_b, t_b = cp_bcomm(X, A0, beta=beta, n_outer=n_outer, eps=eps)
A_j, loss_j, t_j = cp_jcomm(X, A0, beta=beta, n_outer=n_outer, n_inner=n_inner, eps=eps)
```

#### Tucker decomposition (B-CoMM / J-CoMM)

```python
import numpy as np
from comm.models.tucker import tucker_bcomm, tucker_jcomm

X = ...  # ndarray, shape (I1,...,IN), X >= 0
beta = 1.5
ranks = (10, 10, 5, 10, 10)  # (J1,...,JN)
eps = 1e-12
n_outer = 100
n_inner = 1

rng = np.random.default_rng(0)
G0 = rng.random(ranks) + 1e-3
A0 = [rng.random((X.shape[n], ranks[n])) + 1e-3 for n in range(X.ndim)]

G_b, A_b, loss_b, t_b = tucker_bcomm(X, G0, A0, beta=beta, n_outer=n_outer, eps=eps)
G_j, A_j, loss_j, t_j = tucker_jcomm(X, G0, A0, beta=beta, n_outer=n_outer, n_inner=n_inner, eps=eps)
```

#### Plotting convergence (loss vs iteration and loss vs time)

You can plot the optimization history using the helper `comm.utils.plotting.plot_histories`:

```python
from pathlib import Path
from comm.utils.plotting import plot_histories

histories = {
    "B-CoMM": (loss_b / X.size, t_b),
    "J-CoMM": (loss_j / X.size, t_j),
}

plot_histories(
    histories,
    title=f"CP, beta={beta}",
    ylabel="mean loss",
    save_dir=Path("figures"),
    filename="tutorial_cp_beta",
    yscale="log",
    start_iter=1,
    show=True,
)
```

#### Running a snippet directly from terminal (no file)

```bash
python - <<'PY'
import numpy as np
from comm.models.cp import cp_jcomm

X = np.random.rand(20, 18, 16, 14).astype(np.float32)  # replace by your tensor
beta = 1.5
R = 6
rng = np.random.default_rng(0)
A0 = [rng.random((X.shape[n], R), dtype=np.float32) + 1e-3 for n in range(X.ndim)]

A, loss, tt = cp_jcomm(X, A0, beta=beta, n_outer=20, n_inner=1, eps=1e-12)
print("final mean loss:", float(loss[-1] / X.size))
PY
```

## Parameters glossary (common)

- `--beta`: β-divergence parameter ($0 \le \beta < 2$)
- `--iters`: number of **outer** iterations
- `--inner`: number of **inner** iterations for J‑CoMM
- `--eps`: positivity safeguard (clipping floor)
- `--shape`: tensor shape $I_1 I_2 ... I_N$
- `--rank` (CP): CP rank $R$
- `--ranks` / `--tucker_ranks` (Tucker): Tucker multilinear ranks $J_1 ... J_N$
- `--seed`: RNG seed for reproducibility
- `--run_einfact`: enable competitor (requires `pip install -e ".[competitors]"`)
- `--einfact_path`: path to `einfact.py` (vendor file)
- `--device`: `cpu` (or `cuda` if available and supported)

## Notes on loss scaling (important)

- `comm.core.beta_divergence` returns a **sum** over all entries.
- NNEinFact reports a **mean** loss.

All benchmark scripts compare methods using **mean loss** by dividing our summed loss by `X.size` before plotting.

## Project structure

```
comm_project/
  src/comm/
    core.py                 # beta divergence, cached einsum, MU helpers
    models/
      cp.py                 # CP: B-CoMM, J-CoMM, MU-unfold baseline
      tucker.py             # Tucker: B-CoMM, J-CoMM, MU-unfold baseline
    scripts/
      bench_cp.py
      bench_tucker.py
      bench_real_uber.py
      bench_apples.py       # thread-controlled, time-to-target
      bench_sweep.py        # multi-seed averaging + torch_threads overlays
    competitors/
      einfact_wrapper.py    # NNEinFact wrapper (patched + exact init injection)
    utils/
      plotting.py
  data/
  figures/
  einfact.py                # vendor copy of Hood & Schein's reference code
  pyproject.toml
  README.md
  LICENSE
```

## 📚 References

- **NNEinFact**: John Hood and Aaron Schein (2026). Near-Universal Multiplicative Updates for Nonnegative Einsum Factorization. arXiv preprint arXiv:2602.02759.

## License

MIT License — see `LICENSE`.

## 📖 How to Cite

If you use this code in academic work, please cite the accompanying paper draft 

```
@misc{leplat2026jointmajorizationminimizationnonnegativecp,
      title={Joint Majorization-Minimization for Nonnegative CP and Tucker Decompositions under $\beta$-Divergences: Unfolding-Free Updates}, 
      author={Valentin Leplat},
      year={2026},
      eprint={2602.14683},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2602.14683}, 
}
```

and include a reference to this repository.

## 📧 Support and Contact

For questions, bug reports, or contributions, please contact:
**v dot leplat [at] innopolis dot ru**

