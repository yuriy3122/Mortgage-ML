<img width="600" alt="kan_plot" src="https://github.com/yuriy3122/Mortgage-ML/blob/main/kan.png">

# Mortgage-ML

This code is an example of usage of Kolmogorov-Arnold Networks (KANs) for binary classification task<br>
Based on the github repo: <a href="url">https://github.com/KindXiaoming/pykan</a><br>
KAN model used to estimate the creditworthiness of loan and credit applicants using historical data in the context of macroeconomic factors.<br>
Historical data based on FFIEC Home Mortgage Disclosure Act <a href="url">https://ffiec.cfpb.gov/</a>

Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better than MLPs in terms of both model **accuracy** and **interpretability**.

## Installation
Pykan can be installed via PyPI or directly from GitHub.

**Pre-requisites:**

```
Python 3.12 or higher
pip
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```
## Efficiency mode
For many machine-learning users, when (1) you need to write the training loop yourself (instead of using ``model.fit()``); (2) you never use the symbolic branch, it is important to call ``model.speed()`` before training! Otherwise, the symbolic branch is on, which is super slow because the symbolic computations are not parallelized!

