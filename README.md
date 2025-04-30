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
```
Using data scaling to get good results:
```
scaler = preprocessing.MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
X = pd.DataFrame(x_scaled)
```
Splitting data into train val and test and then convert them to Torch tensors:
```
# Splitting data to train val test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=5)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=5)

# Converting data to Torch tensor
train_input = torch.tensor(X_train.to_numpy(), dtype=torch.float32, device=device)
train_label = torch.tensor(y_train.to_numpy(), dtype=torch.long, device=device)
val_input = torch.tensor(X_val.to_numpy(), dtype=torch.float32, device=device)
val_label = torch.tensor(y_val.to_numpy(), dtype=torch.long, device=device)
test_input = torch.tensor(X_test.to_numpy(), dtype=torch.float32, device=device)
test_label = torch.tensor(y_test.to_numpy(), dtype=torch.long, device=device)
```
This dictionary is necessary for KAN training:
```
dataset = {
    'train_input': train_input,
    'train_label': train_label,
    'val_input': val_input,
    'val_label': val_label,
    'test_input': test_input,
    'test_label': test_label
}
```

## KAN model building
```trainModel.py
model = KAN(width=[14, 5, 2], grid=10, k=3).to(device)
```
First hyperparameter is "width" which defines the structure of model.
14 means 14 feature, 5 hidden neurons and 2 output edge for binary classification.
"grid" parameter refer to the number of combined points of each functional section.
KAN tries to complete non-linear relationships by processing data on this grid.
The k parameter determines the maximum degree of basic functions.

## KAN model training
```
results = model.fit({'train_input': train_input, 'train_label': train_label, 'test_input': val_input, 'test_label': val_label},
                     metrics=(train_acc, test_acc), opt="LBFGS", steps=100, loss_fn=torch.nn.CrossEntropyLoss())
```
"opt" hyperparameter could be “LBFGS” or “Adam”

"steps" parameter specifies the total number of iterations to be performed during the training process. 
This is similar to epoch in some ways, but each 'step' usually runs on a batch, 
and an epoch means processing the entire data set once.

"loss_fn" for binary classification task this is CrossEntropLoss()

## KAN model results

KAN Model accuracy:<br>
Train ACC: 0.943069885471969<br>
Val ACC: 0.9375194764724213<br>
Test ACC: 0.9370520411343097<br>

MLP (Multilayer perceptron) accuracy on the same dataset:
Train ACC: 0.8232<br>

## Plotting KAN network
model.plot()

<img width="600" alt="kan_plot" src="https://github.com/yuriy3122/Mortgage-ML/blob/main/KAN-scheme.png">

<br>

<img width="600" alt="kan_plot" src="https://github.com/yuriy3122/Mortgage-ML/blob/main/plot.png">





