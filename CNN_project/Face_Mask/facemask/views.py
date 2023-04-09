from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import PhotoForm
from .models import Facemask
from django.utils import timezone

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
from glob import glob
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model


def check_mask(request):
    if request.method == 'POST':
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            result = Facemask.objects.latest('id')
            
            fileName = str(result.photo)
            # images/540.png
            image_name = fileName[7:]
            # print(str(result.photo))
            # tempImg = request.FILES['']
            # print("I am resuljsadlfjlas fldskajflajslkd",result.photo)
            datagen = ImageDataGenerator(rescale=1/255)

            CNN_aug_new = Sequential()

            CNN_aug_new.add(Input(shape=(75, 75, 3)))

            #Specify a list of the number of filters for each convolutional layer

            for n_filters in [16,32, 64]:
                CNN_aug_new.add(Conv2D(n_filters,strides=(2, 2), kernel_size=3, activation='relu'))

            # Fill in the layer needed between our 2d convolutional layers and the dense layer
            CNN_aug_new.add(Flatten())

            #Specify the number of nodes in the dense layer before the output
            CNN_aug_new.add(Dense(128, activation='relu'))

            #Specify the output layer
            CNN_aug_new.add(Dense(2, activation='softmax'))
            
            #Compiling the model
            CNN_aug_new.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
            
            CNN_aug_new.load_weights('model_weights.h5')

            img = image.load_img(str(result.photo), target_size=(75, 75))
            img = image.img_to_array(img)
            img = np.array([img])
            # print(img)
            aug_iter = datagen.flow(img, batch_size=1)
            prediction=CNN_aug_new.predict(aug_iter)
            # print(prediction)
            img = image.load_img(str(result.photo),target_size=(75, 75))
            
            if np.argmax(prediction)==0:
                result = "Mask Detected"
                
            else:
                result = "No Mask Detected"
                
            
            context = {'form': form,'result': result, 'image_name': image_name}
            
            return render(request, 'index.html', context)
    else:
        form = PhotoForm()
        result = None
        context = {'form': form, 'result': result}
    return render(request, 'index.html', context)
 
 
