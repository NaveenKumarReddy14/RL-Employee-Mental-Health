
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import warnings
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings("ignore")
data = pd.read_csv(r"/content/sample_data/survey.csv")
data.head()
data.shape
data.isnull().sum()
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100

missing_data_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage (%)": missing_percentage
}).sort_values(by="Percentage (%)", ascending=False)

missing_data_summary
important_columns = [
    "Age", "Gender", "self_employed", "family_history", "work_interfere",
    "no_employees", "tech_company", "benefits",
    "leave","treatment", "remote_work", "mental_health_consequence", "phys_health_consequence", "mental_health_interview"
]

filtered_data = data[important_columns]
filtered_data.head()
filtered_data.isnull().sum()
filtered_data["work_interfere"].value_counts().plot.bar();
filtered_data["self_employed"].value_counts().plot.bar();
filtered_data["work_interfere"] = filtered_data["work_interfere"].fillna("Sometimes")
filtered_data["self_employed"] = filtered_data["self_employed"].fillna("No")
categorical_columns = filtered_data.select_dtypes(include=["object"]).columns
numeric_columns = filtered_data.select_dtypes(include=["int64", "float64"]).columns
from google.colab import drive
drive.mount('/content/drive')
for i in categorical_columns:
    feature = filtered_data[i].unique()
    print(f"{i}: ", feature)
    print('--------------------------------')
    def handle_gender(gender):
    if gender == 'Male':
        return 1
    elif gender == 'male':
        return 1
    elif gender == 'female':
        return 0
    else:
        return 0

filtered_data["Gender"] = filtered_data["Gender"].apply(handle_gender)
def handle_no_employees(no_employees):
    if no_employees == '1-5':
        return 0
    elif no_employees == '6-25':
        return 1
    elif no_employees == '26-100':
        return 2
    elif no_employees == '500-1000':
        return 3
    elif no_employees == 'More than 1000':
        return 4

filtered_data["no_employees"] = filtered_data["no_employees"].apply(handle_gender)
def handle_mental_health_consequence(mental_health_consequence):
    if mental_health_consequence == 'No':
        return 0
    elif mental_health_consequence == 'Yes':
        return 1
    else:
        return 2

filtered_data["mental_health_consequence"] = filtered_data["mental_health_consequence"].apply(handle_mental_health_consequence)

filtered_data["phys_health_consequence"] = filtered_data["phys_health_consequence"].apply(handle_mental_health_consequence)

filtered_data["mental_health_interview"] = filtered_data["mental_health_interview"].apply(handle_mental_health_consequence)
filtered_data["mental_health_consequence"].value_counts()
filtered_data["phys_health_consequence"].value_counts()
filtered_data["mental_health_interview"].value_counts()
print(filtered_data["self_employed"].unique())
print(filtered_data["family_history"].unique())
print(filtered_data["work_interfere"].unique())
print(filtered_data["tech_company"].unique())
print(filtered_data["benefits"].unique())
print(filtered_data["remote_work"].unique())
print(filtered_data["leave"].unique())
X = filtered_data.drop("treatment", axis=1)
y = filtered_data["treatment"].apply(lambda x: 1 if x == "Yes" else 0)
X.to_csv('X.csv')
X.head()
X = pd.get_dummies(X, drop_first=True, dtype=int)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X, y)
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print(accuracy)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X, y)
y_pred = rf.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Random Forest Accuracy: ", accuracy)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_train = np.array(y_train).astype(np.int64)
y_test = np.array(y_test).astype(np.int64)
from sklearn.naive_bayes import GaussianNB

pnn = GaussianNB()
pnn.fit(X_train, y_train)
pnn_pred = pnn.predict(X_test)
pnn_accuracy = accuracy_score(y_test, pnn_pred)
print(f"PNN accuracy: {pnn_accuracy}")
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


input_size = X_train.shape[1]
hidden_size = 16
output_size = 2

model = RNNClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train).unsqueeze(1)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test)

epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    rnn_pred = model(X_test_tensor)
    _, predicted = torch.max(rnn_pred, 1)
    rnn_accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
print(f"RNN accuracy: {rnn_accuracy}")