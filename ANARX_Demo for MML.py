import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import DeepLearner_2 as dl
import os


# U verisi üretimi
t=np.arange(0,501) # Zaman
u=np.zeros(500)
for i in range(500):
    c=np.random.sample(1)
    u[i] = np.sin((0.05*t[i])) + 0.05*c**2
s,u=dl.preprocessing.scale(u.reshape(-1,1))

# yt öğrenme modeli
yt=np.zeros(501)
yt[0]=np.random.sample(1)
yt[1]=np.random.sample(1)
for i in range(499):
    yt[i+2]=0.7653*yt[i+1]-0.231*yt[i]+0.4801*u[i+1]-0.6407*yt[i+1]**2+1.014*yt[i]*yt[i+1]-0.3921*yt[i+1]**2+0.592*yt[i+1]*u[i+1]-0.5611*yt[i]*u[i+1]


yt=yt[:500]
dataset=np.vstack((yt,u.reshape((500,)))).T

def set_shape(dataset,time_steps=4):
    number_of_out=1
    num_rows, num_cols = dataset.shape
    input_data = []
    output_data = []
    for i in range(time_steps, len(dataset)):
        input_data.append(dataset[i-time_steps:i, :])
        output_data.append(dataset[i:(i+number_of_out), 0])
    input_data_Reshaped=np.zeros([num_rows-time_steps,time_steps,num_cols])
    output_data_Reshaped=np.zeros([num_rows-time_steps,1,number_of_out])
    for i in range(len(input_data)):
            input_data_Reshaped[i,:,:] = input_data[i]
            output_data_Reshaped[i,:,:] = output_data[i]        
    print("--------------------------------------------------")
    print("Shape set: Done.")
    return input_data_Reshaped , output_data_Reshaped    

# Train validation veri setlerinin oluşturulması timestep=2
inputs,outputs=set_shape(dataset,time_steps=2)

X_train, X_val, y_train, y_val = train_test_split(inputs[:400], outputs[:400], test_size=0.25)

X_test=inputs[-98:]
y_test=outputs[-98:]

# train_X=train_X.reshape((498,1,4))
inp1=np.expand_dims(X_train[:,0,:],axis=2)
inp2=np.expand_dims(X_train[:,1,:],axis=2)
inp1=inp1.reshape(300,1,2)
inp2=inp2.reshape(300,1,2)
val_inp1=np.expand_dims(X_val[:,0,:],axis=2)
val_inp2=np.expand_dims(X_val[:,0,:],axis=2)
val_inp1=val_inp1.reshape(100,1,2)
val_inp2=val_inp2.reshape(100,1,2)
test_inp1=np.expand_dims(X_test[:,0,:],axis=2)
test_inp2=np.expand_dims(X_test[:,0,:],axis=2)
test_inp1=test_inp1.reshape(98,1,2)
test_inp2=test_inp2.reshape(98,1,2)

# ANARX objesinin tanımlanması
class ANARX():
    def __init__(self, input_shape=(1, 2), num_epochs=50, batch_size=8):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = self.build_model(input_shape)
        self.x_train = None
        self.y_train = None
    
    def build_model(self, input_shape):
        inputA = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Dense(5, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputA)
        x = tf.keras.layers.Dense(1)(x)
        x = tf.keras.Model(inputs=inputA, outputs=x)

        inputB = tf.keras.layers.Input(shape=input_shape)
        y = tf.keras.layers.Dense(5, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputB)
        y = tf.keras.layers.Dense(1)(y)
        y = tf.keras.Model(inputs=inputB, outputs=y)

        combined = tf.keras.layers.add([x.output, y.output])

        model = tf.keras.Model(inputs=[x.input, y.input], outputs=combined)
        return model
    
    def train(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                           loss='mse',metrics=['accuracy'])
        history=self.model.fit(x_train, y_train, validation_data=([x_val[0], x_val[1]],y_val), batch_size=self.batch_size, epochs=self.num_epochs)
        return history
    
    def r_leaky(self,data,alfa):
        for i in range(data.shape[1]):
            if data[0,i] <= 0:
                data[0,i]=data[0,i]/alfa
            else:
                pass
        return data

    def leaky(self,data,alfa=0.01):
        for i in range(data.shape[1]):
            if data[0,i] <= 0:
                data[0,i]=data[0,i]*alfa
            else:
                pass
        return data
    def custom_predict(self, inputs):
        weights = self.model.get_weights()
        x_weights = weights[0]
        x_biases = weights[1]
        y_weights = weights[2]
        y_biases = weights[3]
        z1_weights = weights[4]
        z1_biases = weights[5]
        z2_weights = weights[6]
        z2_biases = weights[7]
        
        x_output = self.leaky((np.dot(inputs[0], x_weights) + x_biases))
        y_output = self.leaky(np.dot(inputs[1], y_weights) + y_biases)
        z1_output = np.dot(x_output, z1_weights) + z1_biases
        z2_output = np.dot(y_output, z2_weights) + z2_biases
        combined = z1_output+z2_output
        return combined
    
    def weights(self):
        weights = self.model.get_weights()
        x_weights = weights[0]
        x_biases = weights[1]
        y_weights = weights[2]
        y_biases = weights[3]
        z1_weights = weights[4]
        z1_biases = weights[5]
        z2_weights = weights[6]
        z2_biases = weights[7]

        return x_weights, x_biases, y_weights, y_biases, z1_weights, z1_biases, z2_weights, z2_biases


# ANARX modelinin oluşturulması
ANARX_Model = ANARX(input_shape=(1, 2))

#ANARX modelinin eğitimi
trained_model=ANARX_Model.train([inp1,inp2], y_train, (val_inp1,val_inp2), y_val)
ANARX_Model.model.save("ANARX_V6.h5")
# ANARX Modeli ile keras kütüphanelerini kullanarak tahmin yapma
pred_train=ANARX_Model.model.predict([inp1,inp2])
pred_val=ANARX_Model.model.predict([val_inp1,val_inp2])
pred_test=ANARX_Model.model.predict([test_inp1,test_inp2])


# ANARX modeli ile bir girdi çifti aracılığıyla custom predict kodu
# pred_custom=ANARX_Model.custom_predict(inputs=[inp1[0],inp2[0]])

# Bütün tahminlerin for döngüsü içinde yapılması
custom_predictions=np.zeros((len(test_inp1),1))
for i in tqdm(range(len(test_inp1))):
    custom_predictions[i,0]=ANARX_Model.custom_predict(inputs=[test_inp1[i],test_inp2[i]])

# Sonuçların grafik üzerinde gösterimi
plt.plot(trained_model.history["loss"])
plt.plot(trained_model.history["val_loss"])
plt.show()

plt.plot(np.arange(0,498),outputs.reshape(-1,1))
# plt.plot(np.arange(0,300),pred_train.reshape(-1,1))
# plt.plot(np.arange(300,400),pred_val.reshape(-1,1))
plt.plot(np.arange(400,498),pred_test.reshape(-1,1))
plt.plot(np.arange(400,498),custom_predictions)
# plt.legend(["Orginal Data", "Train Predictions", "Validation Predictions", "Test Predictions", "ANARX Math Model"],loc="lower left",orientation="horizontal")
plt.show()

# Keras tahminleri ve custom_predictions arasındaki farklar
Residuals=pred_test.reshape(-1,1)-custom_predictions
plt.plot(Residuals)
plt.show()

x_weights, x_biases, y_weights, y_biases, z1_weights, z1_biases, z2_weights, z2_biases= ANARX_Model.weights()

#System model Validation

# V verisi üretimi
v=np.zeros(500)
for i in range(500):
    c=np.random.rand(1)
    v[i] = np.cos((0.1*t[i])) + 0.05*c**2
s,v=dl.preprocessing.scale(v.reshape(-1,1))

ds2_X,ds2_y=set_shape(v.reshape(500,1),time_steps=2)

# yt öğrenme modeli
yt_v=np.zeros(500)
yt_v[0]=np.random.rand(1)
yt_v[1]=np.random.rand(1)
for i in range(498):
    yt_v[i+2]=0.7653*yt_v[i+1]-0.231*yt_v[i]+0.4801*v[i+1]-0.6407*yt_v[i+1]**2+1.014*yt_v[i]*yt_v[i+1]-0.3921*yt_v[i+1]**2+0.592*yt_v[i+1]*v[i+1]-0.5611*yt_v[i]*v[i+1]

yt_v=yt_v[:500]
dataset2=np.vstack((yt_v,v.reshape(500,))).T

inputs_v,outputs_v=set_shape(dataset2,time_steps=2)

inp1_v=np.expand_dims(inputs_v[:,0,:],axis=2)
inp2_v=np.expand_dims(inputs_v[:,1,:],axis=2)
inp1_v=inp1_v.reshape(498,1,2)
inp2_v=inp2_v.reshape(498,1,2)

custom_predictions2=np.zeros((len(inp1_v),1))
for i in tqdm(range(len(inp1_v))):
    custom_predictions2[i,0]=ANARX_Model.custom_predict(inputs=[inp1_v[i],inp2_v[i]])

plt.plot(yt_v)
plt.plot(custom_predictions2)
plt.show()

##############  DENEME2_050423.PY DOSYASI

#trained_model = tf.keras.models.load_model("\ANARX_V6.h5")

#pred_test=np.load("pred_test.npy")
#test_inp1=np.load("test_inp1.npy")
#test_inp2=np.load("test_inp2.npy")

#weights = trained_model.get_weights()

#z2_weights = weights[6]
#z2_biases = weights[7]
#z1_weights = weights[4]
#z1_biases = weights[5]
#y_weights = weights[2]
#y_biases = weights[3]
#x_weights = weights[0]
#x_biases = weights[1]
#inputs=[test_inp1[0], test_inp2[0]]

tahir_sonuc=0.90

u=u.reshape((500,))

def r_leaky(data,alfa):
    for i in range(data.shape[0]):
        if data[i] <= 0:
            data[i]=data[i]/alfa
        else:
            pass
    return data

def leaky(data,alfa):
    for i in range(data.shape[0]):
        if data[i] <= 0:
            data[i]=data[i]*alfa
        else:
            pass
    return data

iter=0
sonuc_deger=[]
sonuc_deger2=[]
u1_deger=[]
u2_deger=[]
res5_list=[]
for i in range(300):
    if iter==0:
        inputs=np.array([u[2],yt[2]]).T.reshape((1,2))
    # elif iter==1:
    #     inputs=np.array([u[3],yt[3]]).T.reshape((1,2))
    # elif iter==2:
    #     inputs=np.array([res5[0,0],yt[2]]).T.reshape((1,2))
    # elif iter==3:
    #     inputs=np.array([res5[0,0],yt[3]]).T.reshape((1,2))
    else:
       inputs=np.array([res5[0,0],sonuc_deger[-1]]).T.reshape((1,2))

    x_output = leaky(np.dot(inputs, x_weights) + x_biases,alfa=0.01)
    z1_output = np.dot(x_output, z1_weights) + z1_biases
    
    res=tahir_sonuc-z1_output
    res1=res-z2_biases
    res2=leaky(np.dot(res1,np.linalg.pinv(z2_weights)),alfa=0.01)
    res3=(res2-y_biases)
    # res4=np.maximum(0,res3)
    res5=np.dot(res3,np.linalg.pinv(y_weights))
    res5_list.append(res5)
    # res2 = np.maximum(0,np.dot(res - z2_biases, np.linalg.pinv(z2_weights)))
    # inputs_1_output = res2 - y_biases
    # res5 = np.dot(inputs_1_output, np.linalg.pinv(y_weights))
    iter=iter+1
    sonuc=(0.7653*res5[0,1]-0.231*inputs[0,1]+0.4801*res5[0,0]-0.6407*res5[0,1]**2+1.014*inputs[0,1]*res5[0,1]-0.3921*res5[0,1]**2+0.592*res5[0,1]*res5[0,0]-0.5611*inputs[0,1]*res5[0,0])
    y_output = leaky(np.dot(res5, y_weights) + y_biases,alfa=0.01)
    z2_output = np.dot(y_output, z2_weights) + z2_biases
    tahir2=z1_output+z2_output
    
    sonuc_deger.append((sonuc))
    sonuc_deger2.append((sonuc))
    u1_deger.append(res5[0,0])
    u2_deger.append(res5[0,1])
    print("Sonuclar : ",iter,'\n',tahir_sonuc,'\n',tahir2)

sonuc=0.7653*res5[0,1]-0.231*inputs[0,1]+0.4801*res5[0,0]-0.6407*res5[0,1]**2+1.014*inputs[0,1]*res5[0,1]-0.3921*res5[0,1]**2+0.592*res5[0,1]*res5[0,0]-0.5611*inputs[0,1]*res5[0,0]
print("Sonuc",sonuc)


#sadece test icin verileri modelde yerine koyduk

res=np.array([0.180511115])
res1=res-z2_biases
res2=leaky(np.dot(res1,np.linalg.pinv(z2_weights)),alfa=0.01)
res3=(res2-y_biases)
# res4=np.maximum(0,res3)
res5=np.dot(res3,np.linalg.pinv(y_weights))
res5_list.append(res5)
print(res5)

y_output = leaky(np.dot(res5, y_weights) + y_biases,alfa=0.01)
z2_output = np.dot(y_output, z2_weights) + z2_biases 
print(y_output)
print(z2_output)
