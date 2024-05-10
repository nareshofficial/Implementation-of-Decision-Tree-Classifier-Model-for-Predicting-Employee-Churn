# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NARESH.P.S
RegisterNumber:  212223040127
*/
import pandas as pd

# Load the dataset
data = pd.read_csv("/content/Employee_EX6.csv")

# Display the first few rows of the dataset
data.head()

# Get information about the dataset
data.info()

# Check for missing values
data.isnull().sum()

# Count the number of employees who left and stayed
data["left"].value_counts()

# Encode the 'salary' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the first few rows of the modified dataset
data.head()

# Select features (independent variables)
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()  # No departments and no left

# Select the target variable (dependent variable)
y = data["left"]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Predict the target variable for the test set
y_pred = dt.predict(x_test)

# Calculate the accuracy of the model
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy

# Predict whether an employee will leave based on given features
dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```

## Output:

### Head:
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/a3acd46e-66c9-43dc-9c85-92cba8c1385f)

### Data.info():
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/12152984-1c9f-4315-a8e7-45b70683916e)

### isnull() and sum():
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/2c6e7a39-6d35-46c4-a4e3-0ae41f344def)

### Data Value Counts():
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/ef5ac630-3bc4-49eb-b6c0-cdd640015ce7)

### Data.head() for salary:
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/dd91d140-e4c7-4d04-bc92-036251d12f55)

### x.head:
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/f7367faa-bc17-4237-a85a-5d5c07f71b87)

### Accuracy Value:
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/65007c77-bb80-4a84-a891-327f6ddbbabf)

### Data Prediction:
![image](https://github.com/23008344/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742655/f3a7586e-af78-4ae2-bf54-4dc09bd44028)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
