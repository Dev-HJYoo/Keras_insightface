import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 0 ~ 255 사이 정수 -> 0 ~ 1 사이 실수로 변환
X_train, X_test = X_train / 255.0, X_test / 255.0

print("X_train shape :", X_train.shape) # X_train shape : (60000, 28, 28)
print("y_train shape :", y_train.shape) # y_train shape : (60000,)
print("X_test shape :", X_test.shape) # X_test shape : (10000, 28, 28)
print("y_test shape :", y_test.shape) # y_test shape : (10000,)

from tensorflow.keras import models, layers

model = models.Sequential()

# 필터 5개, 커널 사이즈 = 3 * 3인 convolution layer - maxpooling 조합
model.add(layers.Conv2D(5, 3, strides = 1, padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# 필터 10개, 커널 사이즈 = 3 * 3인 convolution layer - maxpooling 조합
model.add(layers.Conv2D(10, 3, strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# 1차원으로 변환 후 fc layer 통과(64차원 변환 -> dropout -> 10개 클래스 확률 변환)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation = 'softmax')) # 클래스 10개, softmax 적용하여 각 클래스의 확률로 변환

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 5, batch_size = 16, verbose = 1, validation_data = [X_test, y_test])