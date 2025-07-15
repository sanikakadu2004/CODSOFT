
1. # INSTALL LIBRARIES
!pip install tensorflow numpy pandas matplotlib

2. # IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

3. # LOAD DATASET
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
dataset = pd.read_csv('Iris Plant Dataset.csv')
dataset.head()

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
# Features (sepal length, sepal width, petal length, petal width)
X = iris.data
# Target labels (species)
y = iris.target
# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature names and target names
feature_names = iris.feature_names
target_names = iris.target_names

print("Features:", feature_names)
print("Target classes:", target_names)

4. # IRIS FEATURES BY SPECIES
sns.pairplot(dataset, hue='species')
plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
plt.show()

5. # CORRELATION MATRIX
sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

6. # TRAIN MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("
Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

7. # CONFUSION MATRIX
import matplotlib.pyplot as plt
import seaborn as sns
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
