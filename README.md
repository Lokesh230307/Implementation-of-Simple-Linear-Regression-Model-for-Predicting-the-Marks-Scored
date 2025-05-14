# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data: Read student scores CSV, extract hours (X) and scores (Y).
2. Train Model: Split data into train/test, train Linear Regression model.
3. Predict & Plot: Predict test scores, plot training (orange/red) and test (purple/yellow) data.
4. Evaluate: Compute and print MSE, MAE, RMSE for model performance.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LOKESH S
RegisterNumber:  212224230143
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
#Secgreating Data to variables
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
#splitting train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
Y_pred
#displaying actual values
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ', mae)

rmse=np.sqrt(mse)
print("RMSE = ", rmse)

*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![Screenshot 2025-05-14 094420](https://github.com/user-attachments/assets/dc3d62e6-de8c-4772-a06e-d307a80b471f)
![Screenshot 2025-05-14 094433](https://github.com/user-attachments/assets/fb5692a1-44b9-4227-a18e-fcfdd5b52e54)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
