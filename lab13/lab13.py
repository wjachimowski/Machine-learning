from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import PIL
from tensorflow.keras.layers import Conv2D, Dense,Flatten,MaxPooling2D



(X_train,y_train),(X_test,y_test) = mnist.load_data()
index = np.random.choice(np.arange(len(X_train)),24,replace=False)
figure,axes = plt.subplots(nrows=4,ncols=6,figsize=(16,9))
for item in zip(axes.ravel(),X_train[index],y_train[index]):
    axes,image,target = item
    axes.imshow(image,cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
    plt.tight_layout()

plt.show()

X_train = X_train.reshape((60000,28,28,1))
X_test = X_test.reshape((10000,28,28,1))
X_train= X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

cnn = Sequential()
cnn.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(units=128,activation='relu'))
cnn.add(Dense(units=10,activation='softmax'))
cnn.summary()

image = PIL.Image.open('number.png')
image = np.resize(image,(28,28,1)).reshape(1,28,28,1)

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train,y_train,epochs=5,batch_size=64,validation_split=0.1)
loss,accuracy=cnn.evaluate(X_test,y_test)

print(cnn.predict(image))

