import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Lambda, Add, Concatenate, Dropout, Conv2DTranspose, Reshape

from keras.models import Model, load_model
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import TensorBoard
from scipy import misc
import time
import os

DATA = './Data/'

def mConv(X, filters=8, neck=4, name='None'):
	"""
	Convolutional stack:
		- bottle neck layer to reduce the numbers of parameters
	"""
	Conv1 = Conv2D(filters, (3,3), strides=1, padding='same', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name=name+'_in')(X)
	Conv2 = Conv2D(filters, (3,3), strides=1, padding='same', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name=name+'_out')(Conv1)

	bottle_neck1 = Conv2D(neck, (1,1), strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name=name+'_nout')(Conv2)
	return Conv2, bottle_neck1

def Discriminator(input_shape=None, num_classes=10):
	X_input = Input(input_shape, name='Input')

	X1, bn1 = mConv(X_input, 10, 5, 'Conv_1')
	X1 = MaxPooling2D(2, padding='same')(X1)
	bn1 = MaxPooling2D(2, padding='same')(bn1)

	X2, bn2 = mConv(bn1, 10, 5, 'Conv_2')

	# skip connection to allow gradient to flow more easily
	X3 = Concatenate(name='combine1')([X1, X2])
	X3 = MaxPooling2D(2, padding='same')(X3)

	bn3 = Conv2D(5, (1,1), strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name='fneck')(X3)
	X = Conv2D(10, (3,3), strides=1, padding='same', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name='final')(bn3)

	X = Flatten()(X)
	X = Dropout(0.5)(X)
	
	X = Dense(1, activation='sigmoid', name='is_real')(X)
	model = Model(inputs=X_input, outputs=X, name='Discriminator')

	return model

def Generator(noise_size):
	inp = Input((noise_size,), name='noise')

	X = Dense(60, activation='relu')(inp)
	X = Dropout(0.4)(X)
	# X = Dense(40, activation='relu')(X)
	X = Dense(3*3*15, activation='relu')(X)
	X = Dropout(0.4)(X)
	X = Reshape((3,3,15))(X)
	# X = Conv2DTranspose(20, (3,3), strides=1, padding='same', activation='relu', name='Conv_1')(X)
	X = Conv2DTranspose(15, (3,3), strides=2, padding='valid', activation='relu', name='Conv_2')(X)
	X = Dropout(0.8)(X)
	X = Conv2DTranspose(10, (3,3), strides=2, padding='same', activation='relu', name='Conv_3')(X)
	X = Dropout(0.8)(X)

	X = Conv2DTranspose(1, (3,3), strides=2, padding='same', activation='sigmoid', name='Synthesis_Image')(X)
	model = Model(inputs=inp, outputs=X, name='Generator')
	return model

def load_data():
	data_train = pd.read_csv(os.path.join(DATA, 'train.csv'))
	data_test = pd.read_csv(os.path.join(DATA, 'test.csv'))

	img_rows, img_cols = 28, 28
	input_shape = (img_rows, img_cols, 1)

	X = np.array(data_train.iloc[:, 1:])
	y = to_categorical(np.array(data_train.iloc[:, 0]))

	#Here we split validation data to optimiza classifier during training
	

	#Test data
	X_test = np.array(data_test.iloc[:, :])
	# y_test = to_categorical(np.array(data_test.iloc[:, 0]))
	# print(X_test.shape, X_train.shape)

	X = X.reshape(X.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	

	
	X = X.astype('float32')
	X_test = X_test.astype('float32')
	X /= 255
	X_test /= 255

	return X

def noise_generator(noise_size=10):
	def noise_sampling(batch_size=32):
		return np.random.rand(batch_size, noise_size)
	return noise_sampling


img_shape = (28,28,1)
noise_size = 100
num_classes = 10
batch_size = 50
epochs = 30
lr = 1e-4

#===============
D = Discriminator(img_shape, num_classes)
D.compile(loss='binary_crossentropy',
		optimizer=keras.optimizers.Adam(lr))
D.summary()
#===============
G = Generator(noise_size)
G.summary()
noise = Input((noise_size,))
D.trainable = False
prob = D(G(noise))

C = Model(inputs=noise, outputs=prob, name='Combined')
C.compile(loss='binary_crossentropy',
		optimizer=keras.optimizers.Adam(lr))
C.summary()
#===============
noise_gen = noise_generator(noise_size)

X_train = load_data()
tsize = X_train.shape[0]

Y_train = np.ones((tsize, 1))*0.9

train_time = time.time()
for epoch in range(epochs):
	start_time = time.time()
	X_noise = noise_gen(tsize)
	Y_noise = np.zeros((tsize, 1))

	X_noise_g = noise_gen(tsize)
	Y_noise_g = np.ones((tsize, 1))
	d_loss = []
	g_loss = []
	for i in range(0, tsize, batch_size):
		if (i/batch_size)%100==0:
			print('Training on epoch {} - batch {}'.format(epoch, i/batch_size))
		batch_x = X_train[i:i+batch_size]
		batch_y = Y_train[i:i+batch_size]

		noise_x = X_noise[i:i+batch_size]
		noise_x = G.predict(noise_x)
		noise_y = Y_noise[i:i+batch_size]

		x = np.concatenate([batch_x, noise_x], axis=0)
		y = np.concatenate([batch_y, noise_y], axis=0)

		d_loss.append(D.train_on_batch(x, y))

		x = X_noise_g[i:i+batch_size]
		y = Y_noise_g[i:i+batch_size]

		g_loss.append(C.train_on_batch(x, y))
	print('\tD - {}\n\tG - {}'.format(np.mean(d_loss), np.mean(g_loss)))
	print("Time per epoch: {}".format(time.time()-start_time))
	G.save_weights('./Model_GAN/G_weight_{}'.format(epoch))
	D.save_weights('./Model_GAN/D_weight_{}'.format(epoch))	

	if (epoch+1)%1==0:
		noise = noise_gen(10)
		img = G.predict(noise)
		for j in range(10):
			misc.imsave('./samples_gan/im_{}_{}.png'.format(epoch, j), img[j].reshape(28,28))

print("Total time: {}".format(time.time()-train_time))
G.save_weights('G_weight_gan_final')
D.save_weights('D_weight_gan_final')


# G = Generator(noise_size)
# G.load_weights('G_weight_gan_final')
# # print(G.input)
# n = Input((noise_size,))
# # print(G.inputs)
# nG = Model(inputs=G.input, outputs=G.get_layer('Conv_2').output)
# nG.summary()
