import tensorflow as tf
print(tf.__version__)

# Fetching the dataset, in this case available through the tf.keras.datasets API
mnist = tf.keras.datasets.fashion_mnist

# Load the data in the object into training and test data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Shows how the images are stored.
"""
import matplotlib.pyplot as pyplot
pyplot.imshow(training_images[5])
print(training_labels[5])
print(training_images[5])
"""

# Convert to Normalised values as better suited for CNN
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=25)
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

# Prints the probability of the item being in each of the 10 classes
print(classifications[0])
print(test_labels[0])