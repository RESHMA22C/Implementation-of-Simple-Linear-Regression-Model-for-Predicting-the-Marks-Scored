# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

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
Head Values
 ![image](https://github.com/user-attachments/assets/063501d8-4ed9-47b3-984f-5103ba109a49)
Compare Dataset
   ![image](https://github.com/user-attachments/assets/3b2ccc31-5eb0-403f-9233-832ff21e35f0)
   ![image](https://github.com/user-attachments/assets/8e602af3-64db-4127-b9d5-1b959a9ff9f7)
Predication values of X and Y
    ![image](https://github.com/user-attachments/assets/dda19a47-ec72-4deb-b8ec-b41e2fc2c1d6)
MSE,MAE and RMSE
     ![image](https://github.com/user-attachments/assets/787a6ead-20ea-4494-bc9d-d1e5e7374da1)
Training set
    ![image](https://github.com/user-attachments/assets/b449ac9d-3e91-4abe-a89e-d690ac3b9cf9)
Testing Set
    ![image](https://github.com/user-attachments/assets/c9039680-6492-4e8f-a11f-327807aa4736)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
