# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

#split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

#define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the keras moel
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=10)

#evaluate the keras model
_, accuracy = model.evaluate(X,Y)
print('Accuracy: %.2f' % (accuracy*100))
