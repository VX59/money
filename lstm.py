import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
AMDdata = pd.read_csv("AMD.csv")
GLWdata = pd.read_csv("GLW.csv")
NVDAdata = pd.read_csv("NVDA.csv")
data = AMDdata       
openclose_difference = data['Open'] - data['Close']
highlow_difference = data['High'] - data['Low']
volume = data['Volume']


"""
domain = range(len(highlow_difference))
plt.scatter(domain, highlow_difference, marker='^')
plt.scatter(domain, openclose_difference)
plt.show()
plt.scatter(domain, volume)
plt.show()
"""

# normalize volume with softmax
total = sum(data['Volume'])
normalize = lambda x: (x-min(data["Volume"]))/(max(data["Volume"])-min(data["Volume"]))
softmax_volume = np.array(list(map(normalize, data["Volume"])))

import tensorflow as tf
import tensorflow.keras as ks

features = [data["Close"], softmax_volume, openclose_difference,highlow_difference]
feature_length = len(features)
feature_width = 50
dataset = np.array(features)
dataset = np.transpose(dataset)
r = 0.2
train_features = 600
test_features = 150
split = int(r * len(dataset))
remainder = (len(data)-split) % train_features
train_data = dataset[split:-remainder]
test_data = dataset[:split]
train_close_data = np.array([data['Close'][split:-remainder]])
test_close_data = np.array([data["Close"][:split]])

train_dataset_x = tf.constant(np.reshape(train_data, (int(train_features/feature_width),feature_width,feature_length))[1:])
train_dataset_y = tf.constant(np.roll(np.reshape(train_close_data, (int(train_features/feature_width),feature_width)),1,0)[1:]) # reshape and shift right

test_dataset_x = tf.constant(np.reshape(test_data, (int(test_features/feature_width),feature_width,feature_length))[1:])
test_dataset_y = tf.constant(np.roll(np.reshape(test_close_data, (int(test_features/feature_width),feature_width)),1,0)[1:])

model = ks.Sequential([
    tf.keras.layers.LSTM(356),
    tf.keras.layers.Dense(124),
    tf.keras.layers.Dense(124),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(feature_width)
])
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-3,
            decay_steps=1000,
            decay_rate=0.98,
            staircase=True)

optimizer = tf.keras.optimizers.Adam(lr_schedule)
model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredLogarithmicError(),
            metrics=['accuracy'])

model.fit(train_dataset_x,train_dataset_y, epochs=100, workers=2, validation_split=0.25)

test_index = 0
test = np.reshape(test_dataset_x[test_index], (1,feature_width,feature_length))
prediction = model.predict(test)[0]
domain_test = range(feature_width)

# eliminate outliers
filtered = []
filtered_domain = list(range(len(prediction)))
for i in range(len(prediction)):
    if(prediction[i] >= min(test_dataset_y[test_index]) and prediction[i]*0.95 <= 1.05*max(test_dataset_y[test_index])):
        filtered.append(prediction[i])
    else: 
        filtered_domain.remove(i)
plt.close()
from pprint import pprint
from sympy import Matrix

# fit solution data
b = Matrix([test_dataset_y[test_index]]).transpose()
At = Matrix([domain_test,np.ones((len(domain_test)))])
A = At.transpose()
AtA = At*A
Atb = At*b

augmented = AtA.row_join(Atb)
solutions = augmented.rref()
s = list(solutions[0][:,2].transpose())
s0 = s[0]
s1 = s[1]
def fs(x): return x*s0 + s1

b = Matrix([filtered]).transpose()
At = Matrix([range(len(filtered)),np.ones((len(filtered)))])
A = At.transpose()
AtA = At*A
Atb = At*b

augmented = AtA.row_join(Atb)
solutions = augmented.rref()

x = list(solutions[0][:,2].transpose())
a0 = x[0]
a1 = x[1]
print()
def fp(x): return x*a0 +a1
def fp_match(x): return x*a0 + fs(0)


plt.plot(filtered_domain, filtered, marker="+", color='black', linestyle="dashed")
plt.plot(domain_test, list(map(fp,domain_test)), color='black')
plt.plot(domain_test, list(map(fs,domain_test)), color='green')
plt.plot(domain_test, test_dataset_y[test_index], color='green',  marker="^",linestyle="dashed")

plt.show()
plt.close()
print(a0,a1)
print(s0,s1)

if(a0 > 0.025) : print("sell")
if(a0 < 0.025 and a0 > -0.05): print("hold")
if(a0 < -0.25): print("buy")