import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1 - Importing the data set
google_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = google_dataset.iloc[:, 1:2].values

# Step 2 - Scaling the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Step 2 - Creating a data structure wtih 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Step 3 - Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))

# Step 4 - Building the RNN
# Step 4a - Import the classes
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Step 4b - Adding the LSTM layers and dropout regularization
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# Step 4c - Add the output layer
regressor.add(Dense(units=1))

# Step 5 - Compiling the network
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Step 6 - Fitting the RNN to the training set
regressor.fit(x=X_train, y=y_train, batch_size=32, epochs=100)

# Step 7 - Making the prediction and visualizing the results
# Step 7a - Getting the real stock price of 2017
google_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = google_test.iloc[:, 1:2].values

# Step 7b - Getting the predicted stock price of 2017
total_dataset = pd.concat(objs=(google_dataset['Open'], google_test['Open']), axis=0)
inputs = total_dataset[len(total_dataset)-len(google_test)-60:].values
inputs = inputs.reshape(-1, +1)
inputs_scaled = sc.transform(inputs)
X_test = []
ma_3 = []
for i in range(60, len(inputs_scaled)):
    X_test.append(inputs_scaled[i-60:i, 0])
    ma_3.append(np.mean(inputs_scaled[i-3:i, 0]))
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(x=X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
ma_3 = sc.inverse_transform(np.array(ma_3).reshape(-1, +1))

# Step 7a - Visualizing
plt.plot(real_stock_price, color='r', label='Real Price')
plt.plot(predicted_stock_price, color='b', label='Predicted Price')
plt.plot(ma_3, color='k', label='Moving Average')
plt.title('Google Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()