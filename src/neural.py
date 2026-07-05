import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

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

import tensorflow
