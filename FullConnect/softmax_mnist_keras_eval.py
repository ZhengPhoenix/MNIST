import keras
from keras.datasets import mnist
from keras.models import load_model
import cv2

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_test = keras.utils.to_categorical(y_test, num_classes)

model = load_model('keras_fc.h5')
x = x_test[12, :, :]

result = model.predict(x.reshape(-1, 784))

print('result: {}'.format(result))

cv2.imshow(str(result), x)
cv2.waitKey(0)
cv2.destroyAllWindows()

