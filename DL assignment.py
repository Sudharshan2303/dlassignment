#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[3]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_set = train_datagen.flow_from_directory(
    "D:/Downloads/archive (2)/dataset/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


# In[4]:


test_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_set = test_datagen.flow_from_directory(
    "D:/Downloads/archive (2)/dataset/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)    


# In[5]:


train_set.class_indices


# In[6]:


from tensorflow import keras
cnn=keras.Sequential()
#first con2D
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[224,224,3]))
cnn.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

#second con2D
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))

cnn.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(500,activation='relu'))
cnn.add(keras.layers.Dense(550,activation='relu'))

cnn.add(keras.layers.Dense(750,activation='relu'))
cnn.add(keras.layers.Dense(800,activation='relu'))
cnn.add(keras.layers.Dense(908,activation='relu'))
cnn.add(keras.layers.Dense(1000, activation = 'relu'))
cnn.add(keras.layers.Dense(1,activation='sigmoid'))


# In[7]:


cnn.summary()


# In[8]:


cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')


# In[9]:


model_history=cnn.fit(train_set,validation_data=test_set,epochs=10)


# In[4]:


cnn.evaluate(test_set)





# In[18]:


test = load_img('D:\Downloads\military.jpeg', target_size=(64,64))
test


# In[3]:


from keras.utils import load_img,img_to_array
test=load_img("D:\Downloads\military.jpeg",target_size=(224,224))
test=img_to_array(test)
test=np.expand_dims(test,axis=0)
test.shape


# In[2]:


result=cnn.predict(test)
result


# In[1]:


if result[1][1] ==0:
    prediction='military'
else:
    prediction='other'
print(prediction) 



# In[ ]:




