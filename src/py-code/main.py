import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Read the training data
train_data = pd.read_csv('src/data/train.csv')
print(f"Training data loaded successfully! Shape: {train_data.shape}\n")


print(train_data.head(), "\n")

print("Checking missing values...")
print(train_data.isnull().sum(), "\n")

print("Summary statistics:")
print(train_data.describe(), "\n")

# Feature selection and preprocessing
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
print('Selected features:', features, '\n', 'target:', target, '\n')
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
print('Imputed missing values in age with median, imputed missing values in embarked with mode.\n')
X = train_data[features]
y = train_data[target]
print(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}\n")


numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Define preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created!\n")



print("Building logistic regression model...")

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

model.fit(X, y)
print("Model training complete!\n")

y_pred_train = model.predict(X)
train_acc = accuracy_score(y, y_pred_train)
print(f"Training Accuracy: {train_acc:.4f}\n")

print("Loading test dataset...")
test_data = pd.read_csv('src/data/test.csv')
print(f"Test data loaded successfully! Shape: {test_data.shape}\n")

print("Preprocessing test data...")
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

X_test = test_data[features]

print("Predicting on test data...")
test_predictions = model.predict(X_test)

output = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
output.to_csv("test_predictions-py.csv", index=False)
print("Test predictions saved to 'test_predictions-py.csv'\n")
print("=== SCRIPT COMPLETE ===")