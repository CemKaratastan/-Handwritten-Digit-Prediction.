# Setting Libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Scenario is based on 2 different cases which are classification of new model or old model
# Set parameter as 'True' to classify on a new model
train_new_model = True


if train_new_model:
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    
    #Split datasets as training and testing dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the data (by making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    ####################################### NEURAL NETWORK MODEL CREATION#######################################
    
    # We will define our model sequential
    model = tf.keras.models.Sequential()
    
    # First, add one flattened input layer for the pixels
    # We are using this flatten layer to convert the multidimensional input into one-dimensional, 
    # commonly used in the transition from the convolution layer to the full connected layer.
    model.add(tf.keras.layers.Flatten())
    
    # Second, add two dense hidden layers
    # Dense layer is used for changing the dimension of the vectors by using every neuron.
    # In that way, results from every neuron of the preceding layers go to every single neuron of the dense layer.
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    
    # Then, add dense output layer for the 10 digits
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compile and optimize model with Adam optimizer, use accuracy as a metric
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # -----------> Start to train model
    model.fit(X_train, y_train, epochs=3)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Save the model into directory
    model.save('handwritten_digits.model')
else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.model')

# Load any images which you want to classify in the model and then predict them.
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
