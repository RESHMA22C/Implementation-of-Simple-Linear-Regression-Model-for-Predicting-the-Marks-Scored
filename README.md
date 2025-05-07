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
RegisterNumber:212223040168  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv')

print("Name: Reshma C")
print("Reg No: 212223040168  ")
df.head()

df.tail()

x = df.iloc[:,:-1].values
x

y = df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_pred

y_test

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
 
#Graph plot for training data
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
df.head()

![image](https://github.com/user-attachments/assets/9745e310-d0cd-41d6-86e1-7a0738db7608)

df.tail()

![image](https://github.com/user-attachments/assets/da8d7ce0-5b73-468a-af8a-4f41f99111a9)

Array value of X


![image](https://github.com/user-attachments/assets/9bc8fd7b-6b43-43ea-b3bc-92fa804222a4)

Array value of Y


![image](https://github.com/user-attachments/assets/5ce7877d-6b71-4c44-b70f-e95076ea9fc3)

Values of Y prediction


![image](https://github.com/user-attachments/assets/9157a7bc-7c34-4fd0-8c24-8b572dd13c14)

Array values of Y test


![image](https://github.com/user-attachments/assets/8744c8bd-7c3e-410f-9d7c-e8c91d17085a)

Values of MSE, MAE and RMSE


![image](https://github.com/user-attachments/assets/546c72cf-d8d0-47c7-bc25-bce064d82ecf)

Training Set Graph


![image](https://github.com/user-attachments/assets/c0e6732c-6d7e-4764-a0df-87a55577998d)

Test Set Graph


![image](https://github.com/user-attachments/assets/b9f79dbc-9786-4db5-8afe-22b25f998b68)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
