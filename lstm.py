#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
#load train dataset
train_set = pd.read_csv('./Google_SP_Train.csv')
print("Stock prices data shape:",train_set.shape)
train_set.head()
ts_open_feature = train_set.iloc[:, 1: 2].values
plt.plot(train_set['Open'])
plt.title("Google Stock TS")
plt.xlabel("DAY # ")
plt.ylabel(" Open Price")
plt.show()
# range the values between 0 and 1
scale= MinMaxScaler(feature_range = (0, 1))
ts_open_feature_scaled = scale.fit_transform(ts_open_feature)
X_train = []
y_train = []
# X_train will contain a tuples of 60 values and y_train the 61st value
for i in range(60, len(ts_open_feature_scaled)):
   X_train.append(ts_open_feature_scaled[i-60: i, 0])
   y_train.append(ts_open_feature_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))
lstm_model = Sequential()
#add 1st lstm layer with 80 neurons
lstm_model.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# drop out 20% of units to avoid overfitting 
lstm_model.add(Dropout(rate = 0.2))
#add 2nd lstm layer with 80 neurons
lstm_model.add(LSTM(units = 80, return_sequences = True))
# drop out 20% of units to avoid overfitting 
lstm_model.add(Dropout(rate = 0.2))
#add 3rd lstm layer with 80 neurons
lstm_model.add(LSTM(units = 80, return_sequences = True))
# drop out 20% of units to avoid overfitting 
lstm_model.add(Dropout(rate = 0.2))
#add 4th lstm layer
lstm_model.add(LSTM(units = 80, return_sequences = False))
lstm_model.add(Dropout(rate = 0.2))
##add output layer
lstm_model.add(Dense(units = 1))
# use adam algorithm for gradient descent and MSE for loss function
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# train the model on X_train data with batch size=32 and for 100 epochs to converge
lstm_model.fit(x = X_train, y = y_train, batch_size = 32, epochs = 100)
#load test dataset 
test_set = pd.read_csv('./Google_SP_Test.csv')
# extract open price feature
test_set_open = test_set.iloc[:, 1: 2].values
# show the test dataset shape
print(test_set_open.shape)
# concatenate train and test dataset
all_dataset = pd.concat((train_set['Open'],test_set['Open']), axis = 0)
# consider only the 185 last data
all_datase_2 = all_dataset[len(all_dataset)-len(test_set)- 60: ].values
# scale and reshape the data
all_datase_2 = all_datase_2.reshape(-1, 1)
all_datase_2 = scale.transform(all_datase_2)
# X_test will contain the data to predict
X_test = []
for i in range(60, len(all_datase_2)):
   X_test.append(all_datase_2[i-60:i, 0])
X_test = np.array(X_test)
#reshape test data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predict the 125 values
predict_SP = lstm_model.predict(X_test)
# inverse the predicted values from 0..1 range to initial ranges
predict_SP = scale.inverse_transform(predict_SP)
# plot in red the true values and in blue the predicted ones
plt.plot(test_set_open, color = 'red', label = 'Real stock price')
plt.plot(predict_SP, color = 'blue'  , label = 'Predicted stock price')
plt.title( 'Open stock prices prediction ')
plt.xlabel(' Day#')
plt.ylabel('Open Price')
plt.legend()
plt.show()
