#importing libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score

#data preprocessing

# converting categerical data into integers
data = pd.read_csv("creditcardcsvpresent.csv")
data.Is_declined[data.Is_declined =="Y"] = 1
data.Is_declined[data.Is_declined =="N"] = 0

data.isForeignTransaction[data.isForeignTransaction=="Y"]=1
data.isForeignTransaction[data.isForeignTransaction=="N"]=0

data.isHighRiskCountry[data.isHighRiskCountry=="Y"]=1
data.isHighRiskCountry[data.isHighRiskCountry=="N"]=0

data.isFradulent[data.isFradulent=="Y"]=1
data.isFradulent[data.isFradulent=="N"]=0


x= data.iloc[:,2:10]
y= data["isFradulent"]
y = y.to_frame()


r=[]
for i in range(401,450):
    r.append(i)

train_x = x.iloc[:2800,:]
train_x = train_x.drop(r)
#train_x.values
train_x = train_x.as_matrix()


train_y = y.iloc[:2800,]
train_y = train_y.drop(r)
#train_y.values
train_y = train_y.as_matrix()


df1 = x.iloc[r]
df2 = x.iloc[2800:,:]
test_x = pd.concat([df1,df2])
test_x = test_x.as_matrix()


df3 = y.iloc[r]
df4 = y.iloc[2800:,:]
test_y = pd.concat([df3,df4])
test_y = test_y.as_matrix()


train_x=train_x.astype('float64')
train_y=train_y.astype('float64')
test_y = test_y.astype('float64')
test_x= test_x.astype('float64')


#unsing ANN
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation

#adding input layer and first hidden layer
classifier = Sequential()
classifier.add(Dense(units = 16,activation='relu',kernel_initializer ='uniform',input_dim = 8 ))
#adding second hidden layer
classifier.add(Dense(units = 16,activation='relu',kernel_initializer='uniform'))
#adding output layer
classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))

#compiling layers
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

#fitting ANN to training set
classifier.fit(train_x,train_y,batch_size=10,epochs=250)
#prediction with ANN
pred = classifier.predict(test_x)

pred = (pred>0.5)  #in true or false

#checking accuracy
cm = confusion_matrix(pred,test_y)
print(cm)
acc=accuracy_score(pred,test_y)
print(acc*100)