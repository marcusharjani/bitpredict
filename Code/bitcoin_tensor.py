import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm
import time


# import pandas
import matplotlib.pyplot as plt
# with open('weighted_price.csv', 'rU') as csvfile:
# 	dataset = pandas.read_csv(csvfile, usecols=[1], engine='python', skipfooter=3)
# 	plt.plot(dataset)
# 	plt.show()

# model training
X_train, y_train, X_test, y_test = lstm.load_data('weighted_price.csv', 50, True)
# print X_train


model = Sequential()

model.add(LSTM(
	input_dim=1,
	output_dim=50,
	return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
	100,
	return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
	output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time: ' + str(time.time() - start))

model.fit(
	X_train,
	y_train,
	batch_size=512,
	nb_epoch=1,
	validation_split=0.05)

predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
# lstm.plot_results_multiple(predictions, y_test, 50)
plt.plot(predictions, y_test)
plt.show()




