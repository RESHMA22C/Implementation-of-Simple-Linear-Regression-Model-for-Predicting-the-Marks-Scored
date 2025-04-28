# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import essential libraries for data manipulation, numerical operations, plotting, and regression analysis.

2.Load and Explore Data: Load a CSV dataset using pandas, then display initial and final rows to quickly explore the data's structure. 

3.Prepare and Split Data: Divide the data into predictors (x) and target (y). Use train_test_split to create training and testing subsets for model building and evaluation.
Train Linear Regression Model: Initialize and train a Linear Regression model using the training data.

4.Visualize and Evaluate: Create scatter plots to visualize data and regression lines for training and testing. Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to quantify model performance.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Reshma C
RegisterNumber:  212223040168
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print("Name: Reshma C")
print("Reg no: 212223040168")
df.head()
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE =',mae)

rmse=np.sqrt(mse)
print("RMSE =",rmse)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/3d00c777-ec51-48ae-a3ab-c9e8b1f42116)

![image](https://github.com/user-attachments/assets/df44d748-f92f-44d4-9977-67211a72814a)

![image](https://github.com/user-attachments/assets/3ee6c9ee-cefd-4544-bcc6-6cfe804ac211)

![image](https://github.com/user-attachments/assets/734bc181-9e90-4a48-9fac-0a46dc8847ba)

![image](https://github.com/user-attachments/assets/98ea46e7-da76-4cda-9c9f-54b3f55f0977)

![image](https://github.com/user-attachments/assets/879aa7b2-3f48-4398-8515-9cd675ef103f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
