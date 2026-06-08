import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt



data=pd.read_csv("D:\HTML\ml_prac\datasets\insurance.csv")
print(data.head())
data.info()
print(data.isnull().sum())

#sex-2,smoker-2,region-4
print(data["sex"].unique())
print(data["region"].unique())
#sex,smoker-2=labelencoder;region-4,onehotencoder
en1=LabelEncoder()
data["smoker"]=en1.fit_transform(data["smoker"])
data["sex"]=en1.fit_transform(data["sex"])
print(data.info())

en2=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
one_array=en2.fit_transform(data[["region"]])
one_dataframe=pd.DataFrame(one_array,columns=en2.get_feature_names_out(['region']))
data = pd.concat([data.drop('region', axis=1), one_dataframe], axis=1)
data.info()
print(data.head())
#SPLITTING AND SCALEING
x=data.drop('charges',axis=1)
y=data['charges']
scale1=StandardScaler()
x_scale=scale1.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
mse_test=mean_squared_error(y_test,y_predict)
mae_test=mean_absolute_error(y_test,y_predict)
r2_test=r2_score(y_test,y_predict)
print("mse_test"+":"+str(mse_test))
print("--------------")
print("r2_test"+":"+str(r2_test))
print("--------------------")
print("mae_test"+":"+str(mae_test))
#lets see how the traing data perform
train_pred=model.predict(x_train)
r2_train=r2_score(y_train,train_pred)
print("r2_train"+":"+str(r2_train))

#when we see ouput the r2 score were 78 and 74 lets the model underfit or we will take a complex model 

model2=RandomForestRegressor()
model2.fit(x_train,y_train)
y_predict1=model2.predict(x_test)
mse_test1=mean_squared_error(y_test,y_predict1)
mae_test1=mean_absolute_error(y_test,y_predict1)
r2_test1=r2_score(y_test,y_predict1)
print("mse_test"+":"+str(mse_test1))
print("--------------")
print("r2_test"+":"+str(r2_test1))
print("--------------------")
print("mae_test"+":"+str(mae_test1))
#lets see how the traing data perform
train_pred1=model2.predict(x_train)
r2_train1=r2_score(y_train,train_pred1)
print("r2_train"+":"+str(r2_train1))

plt.bar(["Linear Regression","Random Forest"],[r2_test,r2_test1])
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Comparison")
plt.show()
plt.plot(y_test,y_predict,'o',label="Linear Regression")
plt.plot(y_test,y_predict1,'o',label="Random Forest")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")     
plt.legend()
plt.show() 
 








