# import numpy as np

# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, Huber, RobustLoss
# import tensorflow.keras.backend as K

# from tensorflow.keras import layers, regularizers
# from tensorflow.keras.models import Model


# n=10000
# attr = 100
# d = np.random.uniform(low=-1e-3, high=1e-3, size=(n, attr))
# e = np.random.uniform(low=2, high=15.3, size=(n,1))
# X = np.concatenate((d,e),axis=1)
# y = np.random.rand(n, 1)

# model = tf.keras.Sequential([
# 			# layers.BatchNormalization(),
# 		layers.Dense(1024, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),#/rev
# 		layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),#/rev
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),#/rev
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),#/rev
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),#/rev
# 		layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),
# 		layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-6)),#/rev
# 		layers.Dense(1024, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
# 		layers.Dense(1, name="layer5")])

# optimizer = Adam(0.001)

# model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
# model.fit(X, y, validation_split=0.1, epochs=1000, batch_size=24)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


X=np.random.random(size=(20,5))
y=np.random.randint(0,high=5, size=(20,5))

model = Sequential([
            Dense(10, input_dim=X.shape[1]),
            Activation('relu'),
            Dense(5),
            Activation('softmax')
            ])
model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"] )

print('fit model with labels in range 0..5')
history = model.fit(X, y, epochs= 5 )

X = np.vstack( (X, np.random.random(size=(1,5))))
y = np.vstack( ( y, [[8000, 0,0,1,2]]))
print('fit model with labels in range 0..5 plus 8000')
history = model.fit(X, y, epochs= 5 )