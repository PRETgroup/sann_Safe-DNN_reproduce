

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical


#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]

#create model
model = Sequential()

#add model layers
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy as a measure of model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)

model.predict(X_test[:4])

# serialize model to JSON
model_json = model.to_json()
with open("memocode_darknet.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("memocode_darknet.h5")
print("Saved model to disk")
