import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.util import invert
import cv2
from skimage.transform import resize
#import Augmentor

image_rows = 512
image_cols = 512

data_path = 'subCT/'
val_path = 'val'
total = 118

def create_data():

	num = 0;
	imgs = np.ndarray((total, image_rows, image_cols), dtype = np.uint8)
	imgs_mask = np.ndarray((total, image_rows, image_cols), dtype = np.uint8)

	print('-'*30)
	print('Creating raw training images...')
	print('-'*30)
	train_path = os.path.join(data_path + '/test')
	aug_path = os.path.join(data_path + '/test/output')
	folders = os.listdir(train_path)
	#aug_folders = os.listdir(aug_path)
	for image_name in folders:
		if 'output' in image_name:
			continue
		img = cv2.imread(os.path.join(data_path + '/test/', image_name), 0)
		img_mask = cv2.imread(os.path.join(data_path + '/gt/', image_name.split('.')[0] + '.bmp'), 0)
		
		if img.shape != (512, 512):
			img = resize(img, (image_rows, image_cols), preserve_range = True)
			img_mask = resize(img_mask, (image_rows, image_cols), preserve_range = True)

		img = np.array([img])
		img_mask = np.array([img_mask])

		imgs[num] = img
		imgs_mask[num] = img_mask

		if num % 10 == 0:
			print('Done: {0}/{1} images'.format(num, 118))
		num += 1
		'''
	for aug_name in aug_folders:
		if '_groundtruth_(1)_test_' in aug_name:
				continue
		aug_mask_name = "_groundtruth_(1)_test_" + aug_name.split('test_original_')[1]
		aug_img = cv2.imread(os.path.join(data_path + '/test/output', aug_name), 0)
		aug_img_mask = cv2.imread(os.path.join(data_path + '/test/output', aug_mask_name.split('.BMP')[0] + '.bmp'), 0)
			
		
		img = np.array([aug_img])
		img_mask = np.array([aug_img_mask])

		imgs[i] = img
		imgs_mask[i] = img_mask
		if num % 10 == 0:
			print('Done: {0}/{1} images'.format(num, 277))
		num += 1
		'''


	np.save('imgs_train.npy', imgs)
	np.save('imgs_mask_train.npy', imgs_mask)
	print('Saving to .npy files done.')

def load_train_data():
	imgs_train = np.load('imgs_train.npy')
	imgs_mask_train = np.load('imgs_mask_train.npy')
	return imgs_train, imgs_mask_train

def create_val_data():
	val_data_path = os.path.join(val_path)
	images = os.listdir(val_data_path)
	total = len(images)
	imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
	imgs_id = np.ndarray((total, ), dtype=np.int32)
	i = 0
	print('-'*30)
	print('Creating validation data...')
	print('-'*30)
	for image_name in images:
		img_id = image_name.split('.')[0]
		img = cv2.imread(os.path.join(val_path, image_name), 0)
		img = resize(img, (image_rows, image_cols), preserve_range=True)
		img = np.array([img])
		imgs[i] = img
		imgs_id[i] = img_id

		if i % 10 == 0:
			print('Done: {0}/{1} images'.format(i, len(images)))
		i += 1
	print('Loading done.')

	np.save('imgs_val.npy', imgs)
	np.save('imgs_id_val.npy', imgs_id)
	print('Saving to .npy files done.')

def load_val_data():
	imgs_val = np.load('imgs_val.npy')
	imgs_id = np.load('imgs_id_val.npy')
	return imgs_val, imgs_id

if __name__ == '__main__':
	create_data()
	create_val_data()