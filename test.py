import tensorflow.python.keras.backend as K
import tensorflow as tf
import numpy as np

x = K.variable(np.array([[1, 2], [3, 4]]))
y = K.variable(np.array([[1, 2], [3, 4]]))
xy = K.dot(x, y)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(xy))
