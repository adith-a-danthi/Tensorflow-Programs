import tensorflow as tf

# To stop training after 99% accuracy is reached
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

callback = myCallback
mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, callbacks=callback)

op = model.predict(test_images)

print(len(op), "test images")
n = int(input("Enter image number to check prediction: "))

print(op[n])
print(test_labels[n])