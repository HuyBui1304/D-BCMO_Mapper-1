# D_BCMO-Mapper

**D_BCMO-Mapper** is a metaheuristic framework for automatic hyperparameter optimization in Topological Data Analysis (TDA). It combines the **D-Mapper** algorithm (distribution-guided Mapper) with **BCMO** (Balancing Composite Motion Optimization) to automatically tune the `(n_cubes, alpha)` parameters of the Mapper pipeline.

> Bui et al., "D_BCMO-Mapper: A Metaheuristic Framework for Distribution-Guided Parameter Optimization in Topological Data Analysis," under review at *Vietnam Journal of Computer Science* (2026).

---

## Overview

The standard [Mapper algorithm](https://kepler-mapper.scikit-tda.org/) requires manual selection of cover parameters. D_BCMO-Mapper automates this via:

1. **D-Mapper** ([Tao & Ge, 2025](https://doi.org/10.1186/s12859-025-06085-5) · [GitHub](https://github.com/ShufeiGe/D-Mapper)) — replaces the uniform cover with a GMM-based distribution-guided cover (`D_Cover`), giving more adaptive interval construction.
2. **BCMO** ([Le-Duc et al., 2020](https://doi.org/10.1016/j.ins.2020.02.013) · [GitHub](https://github.com/ThangLe-duc/BCMO_Package)) — metaheuristic optimizer that searches for the `(n_cubes, alpha)` pair maximizing the adjusted silhouette coefficient `SC_adj`.

The objective `SC_adj = w1 * SC_norm + w2 * TSR` balances cluster quality (normalized silhouette) and topological robustness (Topological Signal Rate).

---

## Files

| File | Description | Origin |
|---|---|---|
| `dcover.py` | `D_Cover` — GMM-based distribution-guided cover | D-Mapper (Tao & Ge 2024) |
| `dmapper.py` | `D_Mapper` — Mapper pipeline using `D_Cover` | D-Mapper (Tao & Ge 2024) |
| `evaluate.py` | `compute_SC_adj` — adjusted silhouette metric | D-Mapper (Tao & Ge 2024) |
| `bcmo.py` | `BCMO` — Balancing Composite Motion Optimization | This work |
| `experiment.py` | `auto_seed`, `dmapper_objective`, `run_bcmo` — notebook helpers | This work |

---

## Installation

### 1. Install KeplerMapper

```bash
pip install kmapper
```

### 2. Add D_BCMO-Mapper modules to kmapper

Find the kmapper package directory:

```bash
python -c "import kmapper; import os; print(os.path.dirname(kmapper.__file__))"
```

Copy the five `.py` files into that directory:

```
dcover.py
dmapper.py
evaluate.py
bcmo.py
experiment.py
```

### 3. Register the new modules in `__init__.py`

Open `kmapper/__init__.py` and add at the end:

```python
from .dcover import D_Cover
from .dmapper import D_Mapper
from .evaluate import compute_SC_adj
from .bcmo import BCMO
from .experiment import auto_seed, dmapper_objective, run_bcmo
```

### 4. Install additional dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
import kmapper as km
import numpy as np
from sklearn import cluster

# 1. Load data
X = np.loadtxt("your_data.csv", delimiter=",")

# 2. Lens (projection)
mapper = km.D_Mapper(verbose=0)
lens = mapper.fit_transform(X)

# 3. Define the objective function
@km.dmapper_objective
def objective(n, a):
    cover = km.D_Cover(n_cubes=n, alpha=a)
    graph = mapper.map(lens, X, cover=cover,
                       clusterer=cluster.DBSCAN(eps=0.5, min_samples=3))
    return km.compute_SC_adj(X, lens, graph, cover, type='d')

# 4. Run BCMO to find optimal (n_cubes, alpha)
km.run_bcmo(
    objective_fn=objective,
    bounds=[(5, 15), (0.01, 0.2)],
    NP=20,
    gen=10,
    run_id=1,
    output="results_BCMO.txt",
)
```

---

## Example Notebooks

The `implement/` folder contains one Jupyter notebook per dataset. Each notebook runs the BCMO search and builds the graph at the best-found parameters:

| Dataset | Folder | Data | BCMO bounds `(N, α)` |
|---|---|---|---|
| Disjoint circles | `two_cir/` | Simulated: [D-Mapper repository](https://github.com/ShufeiGe/D-Mapper/tree/main/dmapper_examples/two_cir) | (10, 12), (0.01, 0.20) |
| Intersecting circles | `two_i_cir/` | Simulated: [D-Mapper repository](https://github.com/ShufeiGe/D-Mapper/tree/main/dmapper_examples/two_i_cir) | (6, 8), (0.01, 0.10) |
| Trefoil knot | `trefoil/` | Simulated: [`make_trefoil`](https://github.com/braindynamicslab/dyneusr/blob/master/dyneusr/datasets/trefoil.py) from DyNeuSR | (7, 9), (0.01, 0.10) |
| Cat | `cat/` | [Cat poses](https://people.csail.mit.edu/sumner/research/deftransfer/data.html) (Sumner & Popović, 2004) | (7, 15), (0.01, 0.10) |
| Lion | `lion/` | [Lion poses](https://people.csail.mit.edu/sumner/research/deftransfer/data.html) (Sumner & Popović, 2004) | (8, 10), (0.01, 0.10) |
| Horse | `horse/` | [Horse poses](https://people.csail.mit.edu/sumner/research/deftransfer/data.html) (Sumner & Popović, 2004) | (5, 12), (0.01, 0.10) |
| Human | `human/` | [Mesh Segmentation Benchmark](https://segeval.cs.princeton.edu/), model `1.off` (Chen et al., 2009) | (5, 12), (0.01, 0.10) |
| SARS-CoV-2 | `covid-19/` | Sequences: [GenBase](https://ngdc.cncb.ac.cn/genbase/); distance matrix: [D-Mapper repository](https://github.com/ShufeiGe/D-Mapper/tree/main/dmapper_examples/covid-19) | (10, 20), (0.001, 0.007) |

The input files are already included in each folder, so these links are only needed to trace or re-download the original data:

- **Trefoil knot** has no data file: it is generated inside the notebook with DyNeuSR. Install it with `pip install seaborn "nilearn<0.13" "ipython<9"` followed by `pip install --no-deps dyneusr`, because a plain `pip install dyneusr` downgrades kmapper to 1.2.0 and `D_Mapper` can no longer be imported.
- **Cat, Lion, Horse** use only the reference mesh of each pose set: `cat-reference.obj`, `lion-reference.obj` and `horse-reference.pose` in the original archives. The `lion/` and `horse/` folders store the vertices as `.csv`.
- **SARS-CoV-2** reads the precomputed distance matrix `dis_matrix.csv` (357 sequences), not the raw sequences.

Each notebook's first cell sets up the path. After completing the installation steps above, run the notebooks from within their respective folders.

---

## Key Parameters

### `D_Cover`

| Parameter | Default | Description |
|---|---|---|
| `n_cubes` | `None` | Number of hypercubes per dimension |
| `alpha` | `None` | Symmetrical quantile level controlling interval width |
| `tol` | `0.75e-3` | GMM convergence threshold; decrease for more stable fits |
| `dis` | `scipy.stats.norm` | Distribution family for interval construction |

### `compute_SC_adj`

| Parameter | Default | Description |
|---|---|---|
| `type` | required | `'d'` for D-Mapper, `'k'` for standard KeplerMapper |
| `N` | `100` | Bootstrap iterations for topological robustness |
| `w1`, `w2` | `0.5`, `0.5` | Weights for SC_norm and TSR |

### `run_bcmo`

| Parameter | Default | Description |
|---|---|---|
| `bounds` | required | List of `(min, max)` for `[n_cubes, alpha]` |
| `NP` | `20` | Population size |
| `gen` | `10` | Number of generations |
| `output` | `None` | Path to append-only results log (`.txt`) |

---

## Attribution and Licenses

### D-Mapper (dcover.py, dmapper.py, evaluate.py)

These files are derived from **D-Mapper v1.0** by Yuyang Tao and Shufei Ge.

> Yuyang Tao, Shufei Ge. "A distribution-guided Mapper algorithm." *arXiv:2401.12237*, 2024.

```
D-Mapper v1.0. Copyright (c) 2024. Yuyang Tao and Shufei Ge.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS ALGORITHM IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### BCMO (bcmo.py)

> T. Le-Duc, Q.-H. Nguyen, H. Nguyen-Xuan. "Balancing composite motion optimization." *Information Sciences*, 520, 250–270, 2020.

### KeplerMapper

These modules are designed as extensions to [KeplerMapper](https://kepler-mapper.scikit-tda.org/) (BSD 3-Clause License).

### Example datasets

- **Cat, Lion, Horse:** the mesh data used in this project was made available by Robert Sumner and Jovan Popovic from the Computer Graphics Group at MIT.
- **Human:** model from the [Mesh Segmentation Benchmark](https://segeval.cs.princeton.edu/), distributed under the [AIM@SHAPE General License](https://segeval.cs.princeton.edu/public/AIM.txt), which allows research and non-commercial use only.
- **SARS-CoV-2:** sequences from [GenBase](https://ngdc.cncb.ac.cn/genbase/), National Genomics Data Center, China.

---

## Citation

If you use D_BCMO-Mapper in your research, please cite:

```bibtex
@misc{bui2026dbcmomapper,
  title  = {D\_BCMO-Mapper: A Metaheuristic Framework for Distribution-Guided
            Parameter Optimization in Topological Data Analysis},
  author = {Bui, Quang-Thinh and Ha, Loi and Bui, Minh-Huy and Nguyen, Hung},
  note   = {Under review at Vietnam Journal of Computer Science},
  year   = {2026}
}
```

And the underlying methods:

```bibtex
@article{tao2025distribution,
  title={A distribution-guided Mapper algorithm},
  author={Tao, Yuyang and Ge, Shufei},
  journal={BMC bioinformatics},
  volume={26},
  number={1},
  pages={73},
  year={2025},
  publisher={Springer}
}

@article{le2020balancing,
  title={Balancing composite motion optimization},
  author={Le-Duc, Thang and Nguyen, Quoc-Hung and Nguyen-Xuan, Hung},
  journal={Information Sciences},
  volume={520},
  pages={250--270},
  year={2020},
  publisher={Elsevier}
}
```

And the example datasets:

```bibtex
@article{sumner2004deformation,
  title={Deformation transfer for triangle meshes},
  author={Sumner, Robert W. and Popovi{\'c}, Jovan},
  journal={ACM Transactions on Graphics},
  volume={23},
  number={3},
  pages={399--405},
  year={2004},
  publisher={ACM}
}

@article{Chen2009,
author = {Chen, Xiaobai and Golovinskiy, Aleksey and Funkhouser, Thomas},
title = {A benchmark for 3D mesh segmentation},
year = {2009},
issue_date = {August 2009},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {28},
number = {3},
issn = {0730-0301},
doi = {10.1145/1531326.1531379},
journal = {ACM Transactions on Graphics},
articleno = {73},
numpages = {12},
keywords = {3D mesh analysis, 3D mesh segmentation}
}

@article{cncb2023database,
  title={Database Resources of the National Genomics Data Center, China National Center for Bioinformation in 2023},
  author={{CNCB-NGDC Members and Partners}},
  journal={Nucleic Acids Research},
  volume={51},
  number={D1},
  pages={D18--D28},
  year={2023},
  publisher={Oxford University Press}
}
```

---

## Contact

For bugs or questions related to D_BCMO-Mapper, please open an issue on GitHub or contact: huybm.ds@gmail.com

For questions about the original D-Mapper, contact: taoyy2022@shanghaitech.edu.cn
