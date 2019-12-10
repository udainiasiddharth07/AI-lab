import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler


data=pd.read_csv("C:/Users/User/Downloads/machine_learning-master/DataPreprocessing.csv")
x=data.iloc[:,:3].values
y=data.iloc[:,-1].values
print(x)
print(y)
imputer = Imputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:,1:])
x[:,1:] = imputer.transform(x[:,1:]) 

print("\n")
print(x)

labelEncoder_x=LabelEncoder()
x[:,0]=labelEncoder_x.fit_transform(x[:,0])
print("\n encoded values of x")
print(x)
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
print("\nvalues of x")
print(x)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
print("\nvalues of y")
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
print("\nvalues of trainig set")
print(x_train)
print("\nvalues of testing set")
print(x_test)

