import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from kan import KAN
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score

df = pd.read_csv('train-hmda-data.csv', na_values="Exempt")
df = df.dropna()
X = df[list(df.columns.drop(["action_taken", "loan_amount", "income", "loan_term", "property_value"]))[1:19]]
y = df["action_taken"]
x = X.values

scaler = preprocessing.MinMaxScaler()
scaler.fit(x)

x_scaled = scaler.transform(x)
X = pd.DataFrame(x_scaled)

# Splitting data to train val test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=5)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Converting data to Torch tensor
train_input = torch.tensor(X_train.to_numpy(), dtype=torch.float32, device=device)
train_label = torch.tensor(y_train.to_numpy(), dtype=torch.long, device=device)
val_input = torch.tensor(X_val.to_numpy(), dtype=torch.float32, device=device)
val_label = torch.tensor(y_val.to_numpy(), dtype=torch.long, device=device)
test_input = torch.tensor(X_test.to_numpy(), dtype=torch.float32, device=device)
test_label = torch.tensor(y_test.to_numpy(), dtype=torch.long, device=device)

dataset = {
    'train_input': train_input,
    'train_label': train_label,
    'val_input': val_input,
    'val_label': val_label,
    'test_input': test_input,
    'test_label': test_label
}

model = KAN(width=[14, 5, 2], grid=10, k=3).to(device)

# functions for getting accuracy scores while training
def train_acc():
    predictions = torch.argmax(model(dataset['train_input']), dim=1)
    return torch.mean((predictions == dataset['train_label']).float())

def test_acc():
    predictions = torch.argmax(model(dataset['test_input']), dim=1)
    return torch.mean((predictions == dataset['test_label']).float())

# KAN model training
results = model.fit({'train_input': train_input, 'train_label': train_label, 'test_input': val_input, 'test_label': val_label},
    metrics=(train_acc, test_acc), opt="LBFGS", steps=100, loss_fn=torch.nn.CrossEntropyLoss())

# Predictions of train val and test datasets
test_predictions = torch.argmax(model.forward(test_input).detach(), dim=1)
test_labels = test_label

train_predictions = torch.argmax(model.forward(train_input).detach(), dim=1)
train_labels = train_label

val_predictions = torch.argmax(model.forward(val_input).detach(), dim=1)
val_labels = val_label

# Evaluate metrics
print("Train ACC:", accuracy_score(train_labels.cpu().numpy(), train_predictions.cpu().numpy()))
print("Val ACC:", accuracy_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy()))
print("Test ACC:", accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy()))

# save model to the file
model.to(torch.device("cpu")).saveckpt('./kan-model')

# save scaler to the file
joblib.dump(scaler, 'scaler.pkl')

model.plot()

plt.figure(figsize=(10, 5))
plt.plot(results["train_acc"], label='Training Accuracy')
plt.plot(results["test_acc"], label='Val Accuracy')
plt.plot(results["train_loss"], label='Training Loss')
plt.plot(results["test_loss"], label='Val Loss')
plt.title('Training and Val Accuracy over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Accuracy & Loss')
plt.legend()
plt.grid(True)
plt.show()

y_test = val_labels.cpu().numpy()
y_pred = val_predictions.cpu().numpy()

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

def forward_probabilities(x):
    x = model.forward(x).detach()
    x = torch.nn.functional.softmax(x, dim=1)
    return x

test_proba = forward_probabilities(test_input)[:, 1]
fpr, tpr, _ = roc_curve(test_label, test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve for KAN (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

precision, recall, thresholds = precision_recall_curve(test_label, test_proba)
avg_precision = average_precision_score(test_label, test_proba)

plt.figure()
plt.plot(recall, precision, label=f'KAN (AP={avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()
