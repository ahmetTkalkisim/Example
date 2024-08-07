import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#import DeepLearner_2 as dl
import os
import pandas as pd
import statsmodels.api as sm

dataset=pd.read_excel("\*****.xlsx")

input=np.zeros((12835,7))
output=np.zeros((12835,1))
for i in range(12833):
    input[i,0]=dataset.values[i,9] #Ti
    input[i,1]=dataset.values[i+1,9] #Ti+1
    input[i,2]=dataset.values[i+1,8] #Ph+1
    input[i,3]=dataset.values[i+2,8] #Ph+2
    input[i,4]=dataset.values[i+1,7] #dri+1
    input[i,5]=dataset.values[i+2,7] #dri+2
    input[i,6]=dataset.values[i+2,6] #To+2
    output[i+2,0]=dataset.values[i+2,9]
#output[0,0,0]=dataset.values[i,9]
#output[1,0,0]=dataset.values[i+1,9]
model=keras.Sequential()
model.add(keras.layers.Input(shape=(7,)))
model.add(keras.layers.Dense(7,activation="relu"))
model.add(keras.layers.Dense(7,activation="relu"))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='mse')
X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.30, shuffle=False)

trained_model=model.fit(X_train,y_train,epochs=300,batch_size=8,verbose=2)

predictions=model.predict(input[8982:12833])
plt.plot(y_test.reshape((3851,)), label=('Test Data'))
plt.plot(predictions.reshape((3851,)), label=('ANN Prediction 7-7-1'))
plt.title("ANN Prediction & Test Values")
plt.xlabel("Time")
plt.ylabel("Celcius")
plt.legend()

model.get_weights()
model.save("\\*****.h5")

y_test=y_test.reshape((3851))
predictions=predictions.reshape((3851))
results = np.vstack((y_test, predictions)).T
df_results = pd.DataFrame(data=results, columns=["y_test", "pred"])
df_results.to_excel("\****.xlsx", index=False)
