import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

#doc du lieu tu file csv
Body_df = pd.read_csv("Body_new.txt")                                        #0
Dumbbell_Bicep_Curl_df = pd.read_csv("Dumbbell_Bicep_Curl_new.txt")          #1
Dumbbell_Shoulder_Press_df = pd.read_csv("Dumbbell_Shoulder_Press_new.txt")  #2
Push_Up_df = pd.read_csv("Push Up.txt")                                      #3
Belly_Sticks_df = pd.read_csv("Belly Sticks.txt")                            #4


X = []
y = []
#set so timesteps
num_of_timesteps = 30

#Body_df
dataset = Body_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(num_of_timesteps, n_sample):
    X.append(dataset[i - num_of_timesteps:i,:])
    y.append(0)

#Dumbbell_Bicep_Curl_df
dataset = Dumbbell_Bicep_Curl_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(num_of_timesteps, n_sample):
    X.append(dataset[i - num_of_timesteps:i,:])
    y.append(1)

#Dumbbell_Shoulder_Press_df
dataset = Dumbbell_Shoulder_Press_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(num_of_timesteps, n_sample):
    X.append(dataset[i - num_of_timesteps:i,:])
    y.append(2)

#Push_Up_df
dataset = Push_Up_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(num_of_timesteps, n_sample):
    X.append(dataset[i - num_of_timesteps:i,:])
    y.append(3)

#Belly_Sticks_df
dataset = Belly_Sticks_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(num_of_timesteps, n_sample):
    X.append(dataset[i - num_of_timesteps:i,:])
    y.append(4)

X,y = np.array(X), np.array(y)
print(X.shape, y.shape)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

#train
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2)

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 5, activation= 'softmax'))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")

model.fit(X_train, y_train, epochs=10, batch_size=32 ,validation_data=(X_test, y_test))
model.save("model_new.h5")
