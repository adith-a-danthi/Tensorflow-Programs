import tensorflow as tf
import numpy as np
from tensorflow import keras

# Sequential idicates that layers are defined one after the other but here there is only one layer/dense
# output consists of 1 unit/value (param) and input is also a single unit (param)
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# optimer determines how the next guess should be made
model.compile(optimizer='sgd', loss='mean_squared_error')

# relation y = 3x + 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))
