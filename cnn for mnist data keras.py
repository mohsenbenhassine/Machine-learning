# import libraries
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt 
seed = 7
numpy.random.seed(seed)
# load mnist data and split it into two datasets (train and test) 
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Normalize the pixel values of grayscale images in the range of [0,1].
X_train, X_test = X_train / 255.0, X_test / 255.0
# print the shape of train and test dataset
print('Train dataset shape: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test dataset shape: X=%s, y=%s' % (X_test.shape, y_test.shape))
# plot a 3*3 subplots to get the nine first images in train dataset
rows, cols = 3, 3
fig, ax = plt.subplots(rows, cols)

for i in range(9):
    axi = ax[i// cols, i% cols]
    axi.imshow(X_train[i], cmap=('gray'))
plt.show()



# create model
def cnn_model():
    # create a sequential model: a set of sequential layers from input to output layers
    model = tf.keras.models.Sequential([
    # create a conv. Layer with 32 features ( images with 5*5 sizes) next to the input layer
    tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28, 28,1)),
    # create a pooling layer with 2*2 strides
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    # drop out 20 % of neurons to avoid overfitting  
    tf.keras.layers.Dropout(0.2),
    # create a conv. Layer with 64 features ( images with 5*5 sizes) and use relu active. Function
    tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
    # create a pooling layer with 2*2 strides
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    # create a flatten layer ‘one-dimensional array with shape’ from the previous layer
    tf.keras.layers.Flatten(),


    # create a 128 neurons layer
    tf.keras.layers.Dense(128, activation='relu'),
    # create the output layer composed of 10 classes and using prob. Softmax Function 
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    # compile the model using adam gradient update function and categorical cross entropy function, the accuracy is metric of model performance  
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy'])
    return model
# call of the cnn_model
model = cnn_model()
# print model scheme
model.summary()


# Fit the model

# Train the CNN model with the training data (inputs and labels) and as hyperparameters the batch size=32 and the number of epochs=5 
model.fit(X_train, y_train, epochs=5,batch_size=32)
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Model Error: %.2f%%" % (100*(1-scores[1])))

#plot the first image in test dataset
plt.imshow(X_test[0], cmap=('gray'))
# get the result of the softmax layer in the CNN model
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)
# print the probabilities for each class
print(predictions[0])





