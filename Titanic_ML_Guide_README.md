
# Titanic Survival Prediction: A Machine Learning Approach

## Introduction
This project demonstrates the process of using machine learning to predict survival on the Titanic. It's a step-by-step guide designed for learners new to machine learning, covering data preprocessing, model building, and evaluation using Python.

## Table of Contents
1. Importing Necessary Libraries
2. Loading the Dataset
3. Preprocessing Steps
4. Building the Model
5. Evaluating the Model

## 1. Importing Necessary Libraries
In machine learning with Python, we start by importing the necessary libraries for handling data, preprocessing, and building our model.

**Python Code:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
```

## 2. Loading the Dataset
Loading the data is the first step in our machine learning journey. We use Pandas, a powerful Python library, to load and inspect our dataset.

**Python Code:**
```python
df = pd.read_csv('path/to/titanic.csv')
```

## 3. Preprocessing Steps
Data preprocessing is like getting your ingredients ready before cooking. It involves handling missing data, encoding categorical variables, scaling features, and selecting the most relevant features for our model.

### 3.a) Handling Missing Data
**Python Code:**
```python
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

### 3.b) Encoding Categorical Variables
**Python Code:**
```python
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
```

### 3.c) Feature Scaling
**Python Code:**
```python
scaler = StandardScaler()
df[['Fare', 'Age']] = scaler.fit_transform(df[['Fare', 'Age']])
```

### 3.d) Feature Selection
**Python Code:**
```python
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']
```

### 3.e) Splitting the Data
**Python Code:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 4. Building the Model
We use the RandomForestClassifier for our prediction model. It's a powerful and popular machine learning model suitable for a variety of tasks.

**Python Code:**
```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

## 5. Evaluating the Model
Finally, we evaluate our model's performance by checking its accuracy on the test data.

**Python Code:**
```python
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
```

## Conclusion
This README guides you through the process of using machine learning to predict survival on the Titanic. Each step, from data preprocessing to model evaluation, plays a crucial role in the overall performance of the model.

## Author
Vahid Keshmiri
