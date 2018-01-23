import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
# from keras import layers
# from tensorflow import keras
# from tensorflow.keras import layers
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

sess = tf.Session()
K.set_session(sess)

DATA = './Data/'
MODEL = './Model'
path = os.path.join(MODEL, 'weights.{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5')

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

	return X, y, X_test


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
	
	Category = Dense(num_classes, activation='softmax', name='Category')(X)
	D = Dense(1, activation='sigmoid', name='is_real')(X)
	model = Model(inputs=X_input, outputs=[Category, D], name='Discriminator')

	return model



def Generator(noise_size, num_classes):
	inp1 = Input((noise_size,), name='noise')
	inp2 = Input((num_classes,), name='class')

	inp = Concatenate(name='inp')([inp1, inp2])
	X = Dense(60, activation='relu')(inp)
	# X = Dense(40, activation='relu')(X)
	X = Dense(3*3*15, activation='relu')(X)
	X = Reshape((3,3,15))(X)
	# X = Conv2DTranspose(20, (3,3), strides=1, padding='same', activation='relu', name='Conv_1')(X)
	X = Conv2DTranspose(15, (3,3), strides=2, padding='valid', activation='relu', name='Conv_2')(X)
	X = Conv2DTranspose(10, (3,3), strides=2, padding='same', activation='relu', name='Conv_3')(X)

	X = Conv2DTranspose(1, (3,3), strides=2, padding='same', activation='sigmoid', name='Synthesis_Image')(X)
	model = Model(inputs=[inp1, inp2], outputs=X, name='Generator')
	return model

def Combine(G, D, noise_size, num_classes):
	inoise = Input((noise_size,), name='Noise')
	iclass = Input((num_classes,), name='Class')

	img = G([inoise, iclass])

	D.trainable = False
	category, is_fake = D(img)

	model = Model([inoise, iclass], [category, is_fake], name='Combine')

	return model

def noise_generator(noise_size=10):
	def noise_sampling(batch_size=32):
		return np.random.rand(batch_size, noise_size)
	return noise_sampling

noise_size = 30
num_classes = 10
batch_size = 100
epochs = 500

G = Generator(noise_size, num_classes)
G.summary()

D = Discriminator((28,28,1), num_classes)
D.summary()

D.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
	optimizer=keras.optimizers.Adam(1e-4),
	loss_weights=[1, 1])


C = Combine(G, D, noise_size, num_classes)
C.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
	optimizer=keras.optimizers.Adam(1e-4),
	loss_weights=[1,-1])
C.summary()

X_train, Y_train, X_test = load_data()

noise_gen = noise_generator(noise_size)

# D.load_weights('./Model/D_weight_42')
# G.load_weights('./Model/G_weight_42')

start_training_time = time.time()

for epoch in range(epochs):
	start_time = time.time()
	X_noise = noise_gen(X_train.shape[0])
	Y_noise = to_categorical(np.random.randint(num_classes, size=X_train.shape[0]), num_classes)

	X_noise_g = noise_gen(X_train.shape[0])
	Y_noise_g = to_categorical(np.random.randint(num_classes, size=X_train.shape[0]), num_classes)


	for i in range(0, X_train.shape[0], batch_size):
		if (i/batch_size)%100==0:
			print('Training on epoch {} - batch {}'.format(epoch, i/batch_size))
		batch_x = X_train[i:i+batch_size]
		batch_y = Y_train[i:i+batch_size]
		batch_noise_x = X_noise[i:i+batch_size]
		batch_noise_y = Y_noise[i:i+batch_size]
		batch_noise_x = G.predict([batch_noise_x, batch_noise_y])
		# noise = noise_gen(batch_size)
		# noise_y = to_categorical(np.random.randint(num_classes, size=batch_size), num_classes)

		x = np.concatenate((batch_x, batch_noise_x), axis=0)
		y = np.concatenate((batch_y, batch_noise_y), axis=0)
		is_real = np.concatenate((np.ones(batch_x.shape[0])*0.9,
									np.zeros(batch_noise_x.shape[0])))
		
		D.train_on_batch(x, [y, is_real.reshape(-1,1)])

		noise_x_g = X_noise_g[i:i+batch_size]
		noise_y_g = Y_noise_g[i:i+batch_size]

		C.train_on_batch([noise_x_g, noise_y_g], [noise_y_g, np.zeros(noise_y_g.shape[0])])

	print('Time per epoch {}'.format(time.time()-start_time))
	# if (epoch+1)%5==0:
	G.save_weights('./Model/G_weight_{}'.format(epoch))
	D.save_weights('./Model/D_weight_{}'.format(epoch))	

	# Generate - Image

	if (epoch+1)%5==0:
		noise = noise_gen(10)
		y = to_categorical(np.array(range(10)))
		img = G.predict([noise, y])
		for j in range(10):
			misc.imsave('./samples/im_{}_{}.png'.format(epoch, j), img[j].reshape(28,28))

G.save_weights('G_weight_final')
D.save_weights('D_weight_final')

total_time = time.time() - start_training_time
print('Total time {}'.format(total_time))



# G.load_weights('G_weight')
# noise_gen = noise_generator(10)
# noise = noise_gen(10)
# y = to_categorical(np.array(range(10)))
# img = G.predict([noise, y])
# from scipy import misc
# for i in range(10):
# 	im = img[i]
# 	misc.imsave('im_{}.png'.format(i), im.reshape(28,28))





