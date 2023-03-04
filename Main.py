
import glob
import numpy as np

train_img=glob.glob('./Train/images/*')
train_label=glob.glob('./Train/annots/*')

test_img=glob.glob('./Test/images/*')
test_label=glob.glob('./Test/annots/*')


from preparation_of_data import read
from Generator import ImageGenerator

X_train=[]
Y_train=[]

for i in range(len(train_img)):
    img,label=read(train_img[i],train_label[i])
    X_train.append(img)
    Y_train.append(label)
    
X_train=np.array(X_train)
Y_train=np.array(Y_train)

X_test=[]
Y_test=[]

for i in range(len(test_img)):
    img,label=read(test_img[i],test_label[i])
    X_test.append(img)
    Y_test.append(label)
    
X_test=np.array(X_test)
Y_test=np.array(Y_test)


batch_size = 4
my_training_batch_generator = ImageGenerator(X_train,Y_train,batch_size)
my_validation_batch_generator = ImageGenerator(X_test,Y_test,batch_size)


from tensorflow.keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('Yolo_weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

from utils import yolo_loss
from Yolo_model import YOLO

model=YOLO(448,448,3)
model.summary()

model.compile(loss=yolo_loss ,optimizer='adam')

from custom_LR import CustomLearningRateScheduler,lr_schedule

model.fit(x=my_training_batch_generator,
          steps_per_epoch = int(len(X_train) // batch_size),
          epochs = 60,
          verbose = 1,
          workers= 4,
          validation_data = my_validation_batch_generator,
          validation_steps = int(len(X_test) // batch_size),
           callbacks=[
              CustomLearningRateScheduler(lr_schedule),
              mcp_save
          ])

