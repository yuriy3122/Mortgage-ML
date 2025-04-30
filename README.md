<img width="600" alt="kan_plot" src="https://github.com/yuriy3122/Mortgage-ML/blob/main/kan.png">

# Mortgage-ML

This code is an example of usage of Kolmogorov-Arnold Networks (KANs) for binary classification task.<br>
Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better than MLPs in terms of both model **accuracy** and **interpretability**.

Based on the github repo: <a href="url">https://github.com/KindXiaoming/pykan</a><br>
KAN model used to estimate the creditworthiness of loan applicants using historical data in the context of macroeconomic factors.<br>

## Installation
Pykan library can be installed via PyPI or directly from GitHub.

**Pre-requisites:**

```
Python 3.12 or higher
pip
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```
## Data Preparation
Training dataset based on FFIEC Home Mortgage Disclosure Act: <a href="url">https://ffiec.cfpb.gov/</a>
```
df = pd.read_csv('train-hmda-data.csv', na_values="Exempt")
42785 rows × 20 columns
```

## KAN model building
Here is you can build KAN model.

## KAN model results
MLP compare

## Predictions
You can also plot KAN model.

## Deployment





