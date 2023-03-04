
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from output_layer import Yolo_Reshape
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,Dense,Flatten, Conv2D,GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Concatenate, Activation,Conv2DTranspose, BatchNormalization,Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU



def YOLO(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x=Conv2D(filters=64, kernel_size= (7, 7), strides=(1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(inputs)
    x=LeakyReLU(alpha=0.1)(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

    x=Conv2D(filters=192, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

    x=Conv2D(filters=128, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=256, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

    x=Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

    x=Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same',kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same',kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=1024, kernel_size= (3, 3), strides=(2, 2), padding = 'same')(x)

    x=Conv2D(filters=1024, kernel_size= (3, 3), kernel_regularizer=l2(5e-4))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Conv2D(filters=1024, kernel_size= (3, 3), kernel_regularizer=l2(5e-4))(x)
    #x=LeakyReLU(alpha=0.1)(x)

    x=Flatten()(x)
    x=Dense(800)(x)
    #x=Dense(1024)(x)
    x=Dropout(0.5)(x)
    x=Dense(1470, activation='sigmoid')(x)
    outputs=Yolo_Reshape(target_shape=(7,7,30))(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

