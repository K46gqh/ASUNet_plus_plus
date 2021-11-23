import os
import random
from sklearn.utils import shuffle
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage.filters import median
from skimage.morphology import disk
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, merge, BatchNormalization, Dropout, Add, Activation, Conv2DTranspose, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import objectives
import matplotlib.pyplot as plt
from data import load_train_data, load_val_data
import tensorflow as tf
import time
from adabelief_tf import AdaBeliefOptimizer

#tf.compat.v1.disable_eager_execution()
adb = AdaBeliefOptimizer(learning_rate=3.2e-4, epsilon=1e-14, rectify=False)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 224
img_cols = 224


#LRU = LeakyReLU(alpha = 1/5.5)

def iou(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
	return intersection / union

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def dice_coef_loss(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dce_loss(y_true, y_pred):
	return dice_coef_loss(y_true, y_pred) + K.binary_crossentropy(y_true, y_pred)

def convDBN(num, size, d, x):
	conv = Conv2D(num, (size, size), padding='same', kernel_initializer = 'he_normal')(x)
	conv = BatchNormalization(scale=False, axis=3)(conv)
	conv = Activation('relu')(conv)
	return conv

def normal_block(num, size, d, inputs):
	conv_s = Conv2D(num, (1, 1), padding='same', kernel_initializer = 'he_normal', dilation_rate = 2)(inputs)
	#conv_s = convDBN(num, 1, 1, inputs)
	conv = convDBN(num, size, 1, inputs)
	conv = convDBN(num, size, 1, conv)
	conv = concatenate([conv_s, conv], axis = 3)
	return conv

def get_unet():
	#L1
	inputs = Input((img_rows, img_cols, 1))
	conv00 = normal_block(32, 3, 1, inputs)
	conv00 = BatchNormalization(scale=False, axis=3)(conv00)
	conv00 = Activation('relu')(conv00)
	pool00 = MaxPooling2D(pool_size=(2, 2))(conv00)

	conv10_d = Conv2D(64, (3, 3), dilation_rate = 3, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool00)
	conv10_d = Conv2D(64, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv10_d)
	conv10_d = Conv2D(64, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv10_d)
	conv10_d = BatchNormalization(scale=False, axis=3)(conv10_d)
	conv10_d = Activation('relu')(conv10_d)
	conv10 = normal_block(64, 3, 1, pool00)
	conv10 = concatenate([conv10, conv10_d], axis = 3)
	conv10 = BatchNormalization(scale=False, axis=3)(conv10)
	conv10 = Activation('relu')(conv10)
	pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)

	up01 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv10), conv00], axis=3)
	conv01 = normal_block(32, 3, 1, up01)
	conv01 = BatchNormalization(scale=False, axis=3)(conv01)
	conv01 = Activation('relu')(conv01)

	#L2
	conv20_d1 = Conv2D(128, (3, 3), dilation_rate = 3, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool10)
	conv20_d1 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv20_d1)
	conv20_d1 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv20_d1)
	conv20_d2 = Conv2D(128, (3, 3), dilation_rate = 6, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool10)
	conv20_d2 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv20_d2)
	conv20_d2 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv20_d2)
	conv20_d1 = Activation('relu')(conv20_d1)
	conv20_d2 = Activation('relu')(conv20_d2)
	conv20 = normal_block(128, 3, 1, pool10)
	conv20 = concatenate([conv20, conv20_d1, conv20_d2], axis = 3)
	conv20 = BatchNormalization(scale=False, axis=3)(conv20)
	conv20 = Activation('relu')(conv20)
	pool20 = MaxPooling2D(pool_size=(2, 2))(conv20)

	up11 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv20), conv10], axis=3)
	conv11 = normal_block(64, 3, 1, up11)
	conv11 = BatchNormalization(scale=False, axis=3)(conv11)
	conv11 = Activation('relu')(conv11)

	up02 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv11), conv01, conv00], axis=3)
	conv02 = normal_block(32, 3, 1, up02)
	conv02 = BatchNormalization(scale=False, axis=3)(conv02)
	conv02 = Activation('relu')(conv02) 

	
	#L3
	conv30_d1 = Conv2D(256, (3, 3), dilation_rate = 3, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool20)
	conv30_d1 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv30_d1)
	conv30_d1 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv30_d1)
	conv30_d2 = Conv2D(256, (3, 3), dilation_rate = 12, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool20)
	conv30_d2 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv30_d2)
	conv30_d2 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv30_d2)
	conv30_d3 = Conv2D(256, (3, 3), dilation_rate = 18, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool20)
	conv30_d3 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv30_d3)
	conv30_d3 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv30_d3)
	conv30_d1 = Activation('relu')(conv30_d1)
	conv30_d2 = Activation('relu')(conv30_d2)
	conv30_d3 = Activation('relu')(conv30_d3)
	conv30 = normal_block(256, 3, 1, pool20)
	conv30 = concatenate([conv30, conv30_d1, conv30_d2, conv30_d3], axis = 3)
	conv30 = BatchNormalization(scale=False, axis=3)(conv30)
	conv30 = Activation('relu')(conv30)
	pool30 = MaxPooling2D(pool_size=(2, 2))(conv30)

	up21 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv30), conv20], axis=3)
	conv21 = normal_block(128, 3, 1, up21)
	conv21 = BatchNormalization(scale=False, axis=3)(conv21)
	conv21 = Activation('relu')(conv21) 

	up12 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv21), conv11, conv10], axis=3)
	conv12 = normal_block(64, 3, 1, up12)
	conv12 = BatchNormalization(scale=False, axis=3)(conv12)
	conv12 = Activation('relu')(conv12) 

	up03 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv12), conv02, conv01, conv00], axis=3)
	conv03 = normal_block(32, 3, 1, up03)
	conv03 = BatchNormalization(scale=False, axis=3)(conv03)
	conv03 = Activation('relu')(conv03)
	
	#L4
	conv40_d1 = Conv2D(512, (3, 3), dilation_rate = 3, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool30)
	conv40_d1 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d1)
	conv40_d1 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d1)
	conv40_d2 = Conv2D(512, (3, 3), dilation_rate = 6, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool30)
	conv40_d2 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d2)
	conv40_d2 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d2)
	conv40_d3 = Conv2D(512, (3, 3), dilation_rate = 9, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool30)
	conv40_d3 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d3)
	conv40_d3 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d3)
	#conv40_d4 = Conv2D(512, (3, 3), dilation_rate = 24, padding='same', kernel_initializer = 'he_normal', activation = 'relu')(pool30)
	#conv40_d4 = Conv2D(256, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d4)
	#conv40_d4 = Conv2D(128, (1, 1), padding='same', kernel_initializer = 'he_normal')(conv40_d4)
	conv40_d1 = Activation('relu')(conv40_d1)
	conv40_d2 = Activation('relu')(conv40_d2)
	conv40_d3 = Activation('relu')(conv40_d3)
	#conv40_d4 = Activation('relu')(conv40_d4)
	conv40 = normal_block(512, 3, 1, pool30)
	#conv40 = concatenate([conv40, conv40_d1, conv40_d2, conv40_d3, conv40_d4], axis = 3)
	conv40 = concatenate([conv40, conv40_d1, conv40_d2, conv40_d3], axis = 3)
	conv40 = BatchNormalization(scale=False, axis=3)(conv40)
	conv40 = Activation('relu')(conv40)
	pool40 = MaxPooling2D(pool_size=(2, 2))(conv40)

	up31 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv40), conv30], axis=3)
	conv31 = normal_block(256, 3, 1, up31)
	conv03 = BatchNormalization(scale=False, axis=3)(conv03)
	conv31 = Activation('relu')(conv31) 

	up22 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv31), conv21, conv20], axis=3)
	conv22 = normal_block(128, 3, 1, up22)
	conv22 = BatchNormalization(scale=False, axis=3)(conv22)
	conv22 = Activation('relu')(conv22) 

	up13 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv22), conv12, conv11, conv10], axis=3)
	conv13 = normal_block(64, 3, 1, up13)
	conv13 = BatchNormalization(scale=False, axis=3)(conv13)
	conv13 = Activation('relu')(conv13)

	up04 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv13), conv03, conv02, conv01, conv00], axis=3)
	conv04 = normal_block(32, 3, 1, up04)
	conv04 = BatchNormalization(scale=False, axis=3)(conv04)
	conv04 = Activation('relu')(conv04)
	
	result = Conv2D(1, (1, 1), kernel_initializer = 'he_normal', activation='sigmoid')(conv04)

	model = Model(inputs=[inputs], outputs=[result])
	model.compile(optimizer=adb, loss=dice_coef_loss, metrics=[dice_coef, iou, tf.keras.metrics.MeanSquaredError()])
	#model.compile(optimizer=Adam(3e-4), loss=dice_coef_loss, metrics=[dice_coef, iou, tf.keras.metrics.MeanSquaredError()])
	model.summary()

	return model


def preprocess_train(imgs): #including: median filter, resize
	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype = np.uint8)
	for i in range(imgs.shape[0]):
		img = median(imgs[i], disk(1))
		imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

	imgs_p = imgs_p[..., np.newaxis]
	return imgs_p

def preprocess_mask(imgs):
	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype = np.uint8)
	for i in range(imgs.shape[0]):
		imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

	return imgs_p

def train_and_predict():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, imgs_mask_train = load_train_data()
	imgs_train, imgs_mask_train = shuffle(imgs_train, imgs_mask_train) #shuffle data

	imgs_train = preprocess_train(imgs_train)
	imgs_mask_train = preprocess_mask(imgs_mask_train)

	imgs_train = imgs_train.astype('float32')
	mean = np.mean(imgs_train)  # mean for data centering
	std = np.std(imgs_train)  # std for data normalization

	imgs_train -= mean
	imgs_train /= std

	imgs_mask_train = imgs_mask_train.astype('float32')
	imgs_mask_train /= 255.  # scale masks to [0, 1]

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model = get_unet()
	model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	history = model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=80, verbose=1, shuffle=True, 
		validation_split=0.2, callbacks=[model_checkpoint])

	plt.xticks(np.arange(0, 81, 10), fontsize=15)
	plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
	plt.plot(history.history['dice_coef'])
	plt.plot(history.history['val_dice_coef'])
	#plt.plot(history.history['val_iou'])
	#plt.title('model accuracy')
	plt.ylabel('Dice Coefficient', fontsize=15)
	plt.xlabel('Epoch', fontsize=15)
	plt.legend(['train', 'validation'], loc='lower right', fontsize=15)
	plt.grid(True)
	plt.savefig('learning curve.png')
	plt.clf()

	plt.xticks(np.arange(0, 81, 10), fontsize=15)
	plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
	plt.plot(history.history['dice_coef'])
	plt.plot(history.history['val_dice_coef'])
	#plt.plot(history.history['val_iou'])
	#plt.title('model accuracy')
	plt.ylabel('Dice Coefficient', fontsize=15)
	plt.xlabel('Epoch', fontsize=15)
	#plt.legend(['train', 'validation'], loc='lower right', fontsize=15)
	plt.grid(True)
	plt.savefig('learning curve_2.png')
	plt.clf()

	print('-'*30)
	print('Loading and preprocessing validation data...')
	print('-'*30)

	imgs_val, imgs_id_val = load_val_data()
	imgs_val = preprocess_train(imgs_val)

	imgs_val = imgs_val.astype('float32')
	imgs_val -= mean
	imgs_val /= std

	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	model.load_weights('weights.h5')

	print('-'*30)
	print('Predicting masks on validation data...')
	print('-'*30)
	t = time.time()
	imgs_mask_val = model.predict(imgs_val, verbose=1)
	elapsed = time.time() - t
	print(elapsed)
	np.save('imgs_mask_test.npy', imgs_mask_val)

	print('-' * 30)
	print('Saving predicted masks to files...')
	print('-' * 30)
	pred_dir = 'preds'
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	for image, image_id in zip(imgs_mask_val, imgs_id_val):
		image = (image[:, :, 0] * 255.).astype(np.uint8)
		imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
	train_and_predict()