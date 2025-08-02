🚢 Titanic Survival Prediction
This project uses Logistic Regression to predict passenger survival on the Titanic using machine learning techniques and data exploration in Python.

📁 Dataset
Dataset used: Titanic Dataset (Kaggle)
Alternate Source: Titanic CSV - GitHub

📌 Objectives
Explore the Titanic dataset statistically and visually

Handle missing data and convert categorical data

Build and train a logistic regression model

Evaluate accuracy of predictions

🧪 Libraries Used
python
Copy
Edit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
📊 Data Exploration & Preprocessing
Handled missing values in Age

Converted Sex column to numeric (male=0, female=1)

Selected relevant features: Pclass, Sex, Age, Fare

python
Copy
Edit
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df = df.loc[:, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].copy()
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
📈 Data Visualization
python
Copy
Edit
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Sex")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()
🤖 Model Building
python
Copy
Edit
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
✅ Results
Model accuracy is approximately ~78%

Logistic Regression performs well with basic features

Further improvement possible by adding more features and tuning

📌 Future Improvements
Use more features (e.g. Embarked, SibSp, Parch)

Handle missing Embarked and Cabin values

Try advanced models: Decision Trees, Random Forest, etc.

Hyperparameter tuning

🧠 Concepts Covered
Logistic Regression

Data Cleaning

Feature Engineering

Train/Test Split

Model Evaluation

🔧 How to Run
Open Google Colab

Copy & paste the Python code from this repo

Run the cells to explore the data and build the model
