# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KISHORE KUAMR B
RegisterNumber:212225240073
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear Regression using Gradient Descent
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):

    # Add bias column
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta
    theta = np.zeros((X.shape[1], 1))

    # Gradient Descent
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (
            1 / len(X1) * X.T.dot(errors)
        )

    return theta


# Load dataset
data = pd.read_csv(
    r"/content/50_Startups.csv"
)

# Features
X = data.iloc[:, :-1].values

# Remove categorical column (State)
X = X[:, [0, 1, 2]]

# Target
y = data.iloc[:, -1].values.reshape(-1, 1)

# Scale Feature
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Scale target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

print(X_scaled)

# Train model
theta = linear_regression(X_scaled, y_scaled)

print("Theta")
print(theta)

# New data
new_data = np.array([
    [165349.2, 136897.8, 471784.1]
])

# Scale using training scaler
new_scaled = scaler_X.transform(new_data)

# Add bias
new_scaled = np.c_[
    np.ones((1, 1)),
    new_scaled
]

# Predict
prediction_scaled = new_scaled.dot(theta)

# Convert back to original scale
prediction = scaler_y.inverse_transform(
    prediction_scaled
)

print("Scaled Prediction:", prediction_scaled)
print("Prediction Profit:", prediction)  
*/
```

## Output:
<img width="1920" height="1080" alt="Screenshot 2026-08-17 125944" src="https://github.com/user-attachments/assets/3ca9ecd4-addd-4931-8c96-b92f2f75476e" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
