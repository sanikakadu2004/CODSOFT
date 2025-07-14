
1. # INSTALL LIBRARIES
!pip install pandas scikit-learn matplotlib seaborn

2. # LOAD DATASET
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
df = pd.read_csv('Titanic Dataset.csv')
df.head()

3. # IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
# Check for missing values in the dataset
print("Checking for missing values:")
print(df.isnull().sum())

4. # BASIC EDA
# Handling Data
# Fill missing values in 'Age' with the mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
# Fill missing values in 'Embarked' with the mode (most frequent value)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Check if 'Cabin' column exists before attempting to drop it
if 'Cabin' in df.columns:
    # Drop rows with missing 'Cabin' values as it may not be a critical feature for analysis
    df.drop(columns=['Cabin'], inplace=True)
else:
    print("Column 'Cabin' not found in the DataFrame.")
# Alternatively, fill missing 'Fare' values with the median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Display the cleaned dataset
print(df.head(30))
# Check if any missing values remain
print("
Checking for missing values after handling:")
print(df.isnull().sum())

# Export the cleaned dataset to a new CSV file
cleaned_file_path = 'cleaned_titanic.csv'  # Define the variable with the file path
df.to_csv(cleaned_file_path, index=False)
print(f"
Cleaned data exported to: {cleaned_file_path}")

from google.colab import files
# Download the updated CSV file
files.download('cleaned_titanic.csv')

# Encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

5. # TRAIN MODEL
# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:
", classification_report(y_val, y_pred))

6. # CONFUSION MATRIX
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

7. # CORRELATION MATRIX
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.show()

sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()
