import pandas as pd

# Define column names since iris.data has no header
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Load the dataset from the file
df = pd.read_csv('iris.data', header=None, names=columns)

# Remove empty rows (in case there are blank lines in iris.data)
df = df[df['class'].notna()]

# Print first few rows to verify
print(df.head())
print("Number of rows:", len(df))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load iris dataset from file
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data', header=None, names=columns)

# Drop rows with missing class values (some iris.data files have empty lines)
df.dropna(inplace=True)

# Prepare features and labels
X = df.iloc[:, 0:4].values
y = df['class'].astype('category').cat.codes  # Convert string labels to 0,1,2

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
