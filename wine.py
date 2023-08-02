import matplotlib.pyplot as plt

import pandas as pd

from pandas import concat

import numpy as np

white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';')
 
# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';')

red['type']=1
white['type']=0
# print(red)
# print("+"*40)
# print(white)

wines=pd.concat([red,white],ignore_index=True)


from sklearn.model_selection import train_test_split

x=wines.iloc[:,0:11]
y=np.ravel(wines.type)

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.34,random_state=45)

from keras.models import Sequential,save_model,load_model
from keras.layers import Dense

model=Sequential()

model.add(Dense(16,activation='relu',input_shape=(11,)))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=3,batch_size=1,verbose=1)

y_pred=model.predict(X_test)

print(y_pred)

file_path='./save_model'

save_model(model,file_path)


