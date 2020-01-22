"""The data we'll be using is Open, High, Low, Close, Volume data FOR Bitcoin, Ethereum, Litecoin and Bitcoin Cash.
Data also has a timestamp in UNIX time.
We'd only be focused on the Close and Volume columns. What are these?
The Close column measures the final price at the end of each interval. Here, Interval is of 1min. So, at the end of each minute, what was the price of the asset.
The Volume column is how much of the asset was traded per each interval, in this case, per 1 minute.
In the simplest terms possible, Close is the price of the thing. Volume is how much of thing."""

# We are going to keep past 60mins in the sequence to predict 3 mins into the future.
# Every crypto currency share the same time. So we can relate every crypto with each other with the timestamp.

import pandas as pd 
import time
from sklearn import preprocessing
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
# Since o/p from each layer is i/p for next layer, we can sometimes also normalize that.
# BatchNormalization normalizes the output from each layer and feeds it to the next layer.
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import os
# Suppose you have 100 epochs. The model sometimes reaches good accuracy(local minima), sometimes it reaches global minima, and finally it overfits.
# ModelCheckpoint saves epoch output, whenever the model reaches a local minima.
# You could save every epoch but it'll consume a lot of space. With ModelCheckpoint, we save the best ones.


## df=pd.read_csv("crypto_data/LTC-USD.csv")
## print (df.head())

#STEP 1: WE WILL MAKE A NEW DATAFRAME AND INSERT "CLOSE" AND "VOLUME" OF EVERY CRYPTO IN THAT DATAFRAME WITH TIMESTAMP AS THE INDEX OF THE DATAFRAME

main_df=pd.DataFrame() # Empty dataframe
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 cryptos we want to consider

for ratio in ratios:  # begin iteration to all the crypto
    pathCSV = f"crypto_data/{ratio}.csv"  # get the full path to the csv file the current crypto. (LET BTC)
    # the csv files do not have col names. So we would also need to assign names to all the cols
    df = pd.read_csv(pathCSV, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # BTC csv being read.

    # rename volume and close to include crypto names as well, so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)  # inplace=True means the changes would be made to the
    																						   # current df and we dont need to define new df

    df.set_index("time", inplace=True)  # set time as index, instead of 0,1,2,3, ..., so we can join all cryptos on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    # Now lets store the current crypto (BTC) to our main dataframe (main_df is currently empty for first iteration)

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

#So now we have a main df with time as index, and Close and Volume of every crypto as cols.
# print(main_df)


#STEP 2: Deciding when to BUY or SELL
#This function will take values from 2 columns. 
#If the "future" column is higher, great, it's a 1 (buy). Otherwise it's a 0 (sell). 
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

#To do this, first, we need a future column!

#STEP 3: GETTING FEATURES(current value 'Close') AND LABELS(target: 1=buy, 0=sell)

# To know the Target, we need Future values. Now we want to predict 3 mins in the future. Lets make a 60min in the sequence
FUTURE_PERIOD_PREDICT = 3
SEQ_LEN = 60
RATIO_TO_PREDICT = "LTC-USD"  #Which crypto to predict

# Model Hyperparameters for training the model later:
EPOCHS=10
BATCH_SIZE=64
NAME=f"{SEQ_LEN}-SEQ {FUTURE_PERIOD_PREDICT}-PRED {int(time.time())}-time"

# Now for training, we want a coloumn which has this '3min into the future values'.
# So lets create a new coloumn 'future' and shift every cell of 'Close' , 3 cells up to get future data for current data. 
#We will shift 'Close' of only that crypto which is specified by the user.
#A .shift() will just shift cells for us, a negative shift will shift them "up."
main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)

#STEP 4: MAKE A TARGET COLOUMN WHICH WILL HAVE VALUES 0(SELL) AND 1(BUY) ACCORDING TO THE classify FUNCTION

main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future'])) #map(function name, function parameters)

#print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head())

# STEP 4:TRAIN TEST SPLIT

""" Usually, we randomize the data and take out a validation set which the model hasnt seen and we test on the validation set.
	But in time series, if we take a random validation set from between the dataset, chances are that the values in validation set would come again
	in the data set at some point or the other. So for sequential data and time series, we should always test on future data (i.e. last few % of the data)
	The problem with that method is, the data is inherently sequential, so taking sequences that don't come in the future is likely a mistake. 
	This is because sequences in our case, for example, 1 minute apart, will be almost identical. 
	Chances are, the target is also going to be the same (buy or sell). Because of this, any overfitting is likely to actually pour over 
	into the validation set. Instead, we want to slice our validation while it's still in order. I'd like to take the last 5% of the data. 
	To do that, we'll do:"""

times = sorted(main_df.index.values)  # .value() gets the times in a numpy array. Time (index col) is already sorted, but just to be sure
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # get the time cell value which is at the last 5%

validation_main_df = main_df[(main_df.index >= last_5pct)]  # (because index is sorted.) make the validation data.
main_df = main_df[(main_df.index < last_5pct)]  # now the main_df is all the data up to the last 5%


#STEP 5,6,7: CREATE A FUNCTION TO preprocess (NORMALIZE, CREATE SEQUENCE OF BUY/SELL, AND BALANCE BOTH SEQUENCES)

def preprocess_df(df):
    df = df.drop("future", 1)  # "future" was needed only to find "targer"

#STEP 5: NORMALIZING OUR TRAINING AND TESTING DATA
    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all features ... except for the target (label) itself!
            df[col] = df[col].pct_change()  # part of pandas
            								#.pct_change() usually used in time series & calculates % change b/w the current and a prior element.
            							    # This function by default calculates the percentage change from the immediately previous row.(Part of Pandas)
            							    #pct change "normalizes" the different currencies 
         								    #(each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # .pct_change() can create some NA values in the dataframe
            df[col] = preprocessing.scale(df[col].values)   # part of sklearn
            												# normalizes every feature coloumns between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... just in case. Those nasty NaNs love to creep in.

    #STEP 6: CREATE SEQUENCES

    sequential_data = []  # this is a list that will CONTAIN the sequences of 60min values as feature and one BUY/SELL value as label
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # df.values() would convert df into a numpy array. Then i iterates through that array row wise
        prev_days.append([n for n in i[:-1]])  # store all values except the last one(target)
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # convert the deque in an array and append to sequential data along with the current target
        #So sequential_data is a list of 2 cols. 1st coloumn is the sequence and 2nd coloumn is the target value for the sequence
    random.shuffle(sequential_data)  # shuffle for good measure.

    #STEP 7: BALANCE BOTH BUY/SELL SEQUENCE, SPLIT IN X,y.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    # to balance both buy/sell list, we will find length of shorter list and make both list of the same size.

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    #Lets split the sequence in X and target in y

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), np.array(y)  # return X and y...and make X a numpy array!


train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

# print(f"train data: {len(train_x)} validation: {len(validation_x)}")
# print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
# print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

# STEP 8: BUILDING OUR MODEL

""" we will use 3 layers of lstm, 2 dense layers.
whenever we go from one lstm to another, we always requrn the sequence. But if we go from lstm to dense layer, we dont return the sequence.
Final dense layer would be just of 2 nurons because we have binary choice of BUY/SELL"""

model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2)) # we use this. idk why
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

#Next layer is dense layer. So dont return the sequences.

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

#Configure tensorboard object

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#Above code wouldn't work. Hell No. cz we are on fucking windows. ffs. Well, on Unix, SplitPath splits only on forward slashes; 
#on Windows, it splits on forward slashes unless there are no forward slashes in the string, in which case it splits on backslashes.
#Just use fucking os join to merger paths.
tboard_log_dir = os.path.join("logs",NAME)
tensorboard = TensorBoard(log_dir = tboard_log_dir)

# Configure ModelCheckpoint object. Below 2 code lines work somehow. God knows how. But it does.
# filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=5,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard]
    )

# Score model
# score = model.evaluate(validation_x, validation_y, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])