from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os

os.getcwd()
data=pd.read_csv("musk_csv.csv")
#preprocessing
#droping unnecessary cols
d=data.drop(columns=['ID', 'molecule_name', 'conformation_name'])
#shuffling
shuffle = d.sample(frac=1)


#features and labels
X = d.drop(columns=['class'])
#normalizing
x=preprocessing.normalize(X, norm='l1')
y = d['class']
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2)

#create model
model = Sequential()

#get number of columns in training data
n_cols = x.shape[1]

#add model layers
model.add(Dense(300, activation='relu', input_shape=(n_cols,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore after itiration for 5 times in the dataset
early_stopping_monitor = EarlyStopping(patience=3)
#train model
#h=model.fit(x_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping_monitor])
h=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[early_stopping_monitor])

#F1score
#y_pred = model.predict(x_test, batch_size=64, verbose=1)
#y_pred_bool = np.argmax(y_pred, axis=1)
#print(classification_report(y_test, y_pred_bool))
y_pred1 = model.predict(x_test)
y_pred = np.argmax(y_pred1, axis=1)

# Print f1, precision, and recall scores
print("precision score is:", precision_score(y_test, y_pred , average="macro"))
print("recall score is:", recall_score(y_test, y_pred , average="macro"))
print("f1 score is:", f1_score(y_test, y_pred , average="macro"))



#Plotting
train_loss=h.history['loss']
val_loss=h.history['val_loss']
train_acc=h.history['acc']
val_acc=h.history['val_acc']
xc=range(100)

plt.figure(1,figsize=(7,5))
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.show()