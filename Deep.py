import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split



train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()

test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()

X_train = (train.ix[:,1:].values).astype('float32') # all pixel values
y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')


#Convert train datset to (num_images, img_rows, img_cols) format 
X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(0, 3):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
    
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape



******************************Feature Standardization*****************************************

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px
  
  
 ******************* One Hot encoding of label***********************
    
from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes



**********************Neural Network Architecture**********************************

# fix random seed for reproducibility
seed = 50
np.random.seed(seed)


from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
Lets create a simple model from Keras Sequential layer.


model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)




from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(learning rate=0.01),
 loss='categorical_crossentropy',
 metrics=['accuracy'])


from keras.preprocessing import image
gen = image.ImageDataGenerator()


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=50)
batches = gen.flow(X_train, y_train, batch_size=60)
val_batches=gen.flow(X_val, y_val, batch_size=60)



history=model.fit_generator(batches, batches.n, nb_epoch=2, 
                    validation_data=val_batches, nb_val_samples=val_batches.n)



model.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=60)
history=model.fit_generator(batches, batches.n, nb_epoch=2)








    
