<img width="600" alt="kan_plot" src="https://github.com/yuriy3122/Mortgage-ML/blob/main/kan.png">

# Kolmogorov-Arnold Networks (KANs)

Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better (sometimes much better!) than MLPs in terms of both model **accuracy** and **interpretability**. A quick intro of KANs [here](https://kindxiaoming.github.io/pykan/intro.html).

## Installation
Pykan can be installed via PyPI or directly from GitHub. 

**Pre-requisites:**

```
Python 3.9.7 or higher
pip
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```

**Optional: Conda Environment Setup**
For those who prefer using Conda:
```
conda create --name pykan-env python=3.9.7
conda activate pykan-env
pip install git+https://github.com/KindXiaoming/pykan.git  # For GitHub installation
# or
pip install pykan  # For PyPI installation
```
## Efficiency mode
For many machine-learning users, when (1) you need to write the training loop yourself (instead of using ``model.fit()``); (2) you never use the symbolic branch, it is important to call ``model.speed()`` before training! Otherwise, the symbolic branch is on, which is super slow because the symbolic computations are not parallelized!

