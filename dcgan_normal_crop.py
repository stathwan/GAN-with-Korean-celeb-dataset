
#os.chdir('c:/users/2017B221/desktop/othermodel/dcgan')
import glob
#from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Activation, Dropout
from keras.layers import BatchNormalization, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model

import cv2
import keras.backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
## The GPU id to use, usually either "0" or "1"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
_config = tf.ConfigProto()
_config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=_config))



class DCGAN():
    def __init__(self):
        self.img_rows = 120
        self.img_cols = 96
        self.channels = 3
        
            
        # Following parameter and optimizer set as recommended in paper
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator  = multi_gpu_model(self.discriminator , gpus=2)
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        cut = Input(batch_shape=(64,2))
        img = self.generator(z)
        img = Lambda(self.random_crop, output_shape=(80,80,3))([img,cut])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([z,cut], valid)
        self.combined = multi_gpu_model(self.combined , gpus=2)
        self.combined.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])


    def build_generator(self):

        noise_shape = (100,)

        model= Sequential()
        model.add(Dense(1024*7*6,input_shape=(noise_shape)))
        model.add(Reshape((7,6, 1024)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(512,kernel_size=4,strides=2,activation='relu',padding='same'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(ZeroPadding2D(((0,1),0)))
        model.add(Conv2DTranspose(256 ,kernel_size=5,strides=2,activation='relu', padding="same"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(128,kernel_size=5,strides=2,activation='relu', padding="same"))
        model.add(BatchNormalization(momentum=0.8))
    
        model.add(Conv2DTranspose(3,kernel_size=5,strides=2,activation='tanh', padding="same"))
#        model.add(BatchNormalization(momentum=0.8))
##
#        model.add(Conv2D(3, kernel_size=5, activation='tanh', padding="same"))       
#        model.summary()
        
        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        img_shape=(80,80,3)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))   
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=5, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, kernel_size=5, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(512, kernel_size=5, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
  
        model.add(Conv2D(1024, kernel_size=3, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        img = Input(shape=img_shape)
        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)
        
        return Model(img, valid)
        
    def load_data(self):
        impath=glob.glob('./celeb_sub/*')

#        img_shape = (len(impath)*4,120,96,3)
        img_shape = (len(impath)*8,self.img_rows, self.img_cols, self.channels)
        img_list=np.empty(img_shape)

        for i in range(int(len(impath))) :
            img=cv2.imread(impath[i])
            img = cv2.resize(img, (96, 120))
            
            #simple augmentation
            #flip
            img_aug1= cv2.flip(img, 1)
            
            #mag
            img_row, img_col, img_ch =img.shape
            ratio=0.05
            row_cutoff=round((img_row*ratio/2))
            col_cutoff=round((img_col*ratio/2))
            imp_tmp=img[row_cutoff:img_row-row_cutoff,col_cutoff : img_col-col_cutoff ,0:3]

            img_aug2 = cv2.resize(imp_tmp, (96, 120))
            img_aug3 = cv2.flip(img_aug2, 1)
            
            #
            
            ratio1=0.05
            ratio2=0.1
            row_cutoff=round((img_row*ratio1/2))
            col_cutoff=round((img_col*ratio2/2))
            imp_tmp=img[row_cutoff:img_row-row_cutoff,col_cutoff : img_col-col_cutoff ,0:3]

            img_aug4 = cv2.resize(imp_tmp, (96, 120))
            img_aug5 = cv2.flip(img_aug4, 1)            
            
            ratio1=0.1
            ratio2=0.05
            row_cutoff=round((img_row*ratio1/2))
            col_cutoff=round((img_col*ratio2/2))
            imp_tmp=img[row_cutoff:img_row-row_cutoff,col_cutoff : img_col-col_cutoff ,0:3]

            img_aug6 = cv2.resize(imp_tmp, (96, 120))
            img_aug7 = cv2.flip(img_aug6, 1)             
            
            
            img_list[8*i]=img
            img_list[8*i+1]=img_aug1
            img_list[8*i+2]=img_aug2
            img_list[8*i+3]=img_aug3
            img_list[8*i+4]=img_aug4
            img_list[8*i+5]=img_aug5
            img_list[8*i+6]=img_aug6
            img_list[8*i+7]=img_aug7
            print(i)

        np.random.shuffle(img_list)
        
        
        return(img_list.astype('float32')) 
        
    def load_model(self,path1,path2):
         self.generator.load_weights(path1)
         self.discriminator.load_weights(path2)
        
    def train(self, epochs, batch_size=32, sample_interval=50):

        # Load the dataset
        X_train = self.load_data()

        # Rescale -1 to 1
        X_train = ( X_train.astype(np.float32) -127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):



            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)
            # Train the discriminator
#            d_loss_real = self.discriminator.train_on_batch(imgs, -np.ones((half_batch, 1)))
#            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
#            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)


            #----------------------
            # crop image
            #----------------------
            
            # get random_number 0 to 39
            
            idx_y=round(np.random.normal(20,10))
            if idx_y<0 : idx_y = 0
            if idx_y>40 : idx_y = 40
            
            idx_x=round(np.random.normal(6,4))
            if idx_x<0 : idx_x = 0
            if idx_x>16 : idx_x = 16
            
            cut_idx=np.array([idx_x,idx_y]).reshape(1,2)
            cut_idx=cut_idx.repeat(batch_size,0)

            #crop imgs
            crop_gen_imgs= gen_imgs[:,idx_y:idx_y+80,idx_x:idx_x+80,:]
            crop_imgs = imgs[:,idx_y:idx_y+80,idx_x:idx_x+80,:]

            labels_rf= np.array([[0]*half_batch,[1]*half_batch]).reshape(-1,)
            ImgForTrain=np.concatenate((crop_gen_imgs, crop_imgs))
            d_loss = self.discriminator.train_on_batch(ImgForTrain, labels_rf)            

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.combined.train_on_batch([noise,cut_idx], np.ones((batch_size, 1)))

            # Plot the progress
            print ("{} | G loss: {:0.3f} | D loss: {:0.3f} | D acc: {}".format(epoch, g_loss[0], d_loss[0] ,100*d_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % 5000 == 0 :
                ep=str(epoch).zfill(15)
                self.generator.save_weights('./dcgan_model3/generator{}.hdf5'.format(ep))
                self.discriminator.save_weights('./dcgan_model3/discriminator{}.hdf5'.format(ep))

    def random_crop(self, x):
        img, cut= x[0], x[1]

        xx= K.cast(cut[0,0], 'int32')      
        yy= K.cast(cut[0,1], 'int32')      
        print(xx)
        return img[:,yy:yy+80,xx:xx+80,:]
                
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        digit_size_h=120
        digit_size_w=96
        figure = np.zeros((digit_size_h * r, digit_size_w * c,3))
        k=0
        for i in range(r):
            for j in range(c):
                digit = gen_imgs[k].reshape(digit_size_h, digit_size_w,3)
                figure[i * digit_size_h: (i + 1) * digit_size_h,j * digit_size_w: (j + 1) * digit_size_w,0:3] = digit
                k+=1
#        cv2.imsave('images/mnist_%d.png',figure[...,[2,1,0]])
        ep=str(epoch).zfill(15)
        plt.imsave('./dcgan_images3/celeb_{}.png'.format(ep),figure[...,[2,1,0]])
        
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)

dcgan = DCGAN()
#dcgan.load_model('./saved_model/generator38000.hdf5','./saved_model/discriminator38000.hdf5')
dcgan.train(epochs=1000000000000, batch_size=64, sample_interval=100)


