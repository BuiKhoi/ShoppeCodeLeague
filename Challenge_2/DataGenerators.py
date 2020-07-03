import os
import cv2
import keras
import numpy as np
import imgaug.augmenters as imgaug

class NumpyDataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, folder_link, batch_size=8, shuffle=False):
    self.batch_size = batch_size
    self.folder_link = folder_link
    self.shuffle = shuffle

    self.fetch_data(self.folder_link)

    self.on_epoch_end()
    
  def fetch_data(self, folder_link):
    self.X_data = []
    self.y_true = []
    
    label_counter = 0
    for class_folder in os.listdir(folder_link):
        class_folder_path = folder_link + class_folder + '/'
        for image_file in os.listdir(class_folder_path):
            if '_83.npz' in image_file:
                self.X_data.append(class_folder_path + image_file)
                self.y_true.append(label_counter)
        label_counter += 1
        
    self.X_data = np.array(self.X_data)
    self.y_true = np.array(self.y_true)

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.y_true) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    #print('Index: {}'.format(index))
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#     list_images_temp = [self.images[k] for k in indexes]

    # Generate data
    # X, y = self.__data_generation(list_images_temp)
    X, y = self.__data_generation(indexes)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.y_true))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_idxs):
    '''
    Generate data with the specified batch size
    '''
    # Initialization
    X = np.empty((self.batch_size, 224, 224, 83), dtype=np.uint8)
    y = np.empty((self.batch_size, 42), dtype=np.uint8)

    for i, idx in enumerate(list_idxs):
      temp = np.load(self.X_data[idx])
#       print(self.X_data[idx])
      # X[i,] = cv2.resize(temp['img'].astype(np.float32), (256, 256))
      # y[i,] = cv2.resize(temp['anno'].astype(np.float32), (256, 256))
      X[i,] = temp['data']
      y[i,] = np.zeros((42), np.uint8)
      y[i,][self.y_true[idx]] = 1

    return X, y

class AugDataGenerator(keras.utils.Sequence):
  'Generates data for Keras'

  augs = [
      imgaug.Sequential([imgaug.Affine(scale={"x": (0.75, 1.0), "y": (0.75, 1.0)})]),
      imgaug.Sequential([imgaug.Dropout(0.075)]),
      imgaug.Sequential([imgaug.GaussianBlur(sigma=(0.0, 2.0))]),
      imgaug.Sequential([imgaug.Fliplr(0.5)]),
      imgaug.Sequential([imgaug.Affine(scale=(0.75, 1.0))]),
      imgaug.Sequential([imgaug.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)})]),
      imgaug.Sequential([imgaug.Rotate((-15, 15))]),
      imgaug.Sequential([imgaug.PiecewiseAffine(scale=(0.01, 0.05))]),
      imgaug.Sequential([imgaug.PerspectiveTransform(scale=(0.01, 0.15))])
    ]

  def __init__(self, folder_link, batch_size=8, shuffle=False):
    self.batch_size = batch_size
    self.folder_link = folder_link
    self.shuffle = shuffle

    self.fetch_data(self.folder_link)

    self.on_epoch_end()
    
  def fetch_data(self, folder_link):
    self.X_data = []
    self.y_true = []
    
    label_counter = 0
    for class_folder in os.listdir(folder_link):
        class_folder_path = folder_link + class_folder + '/'
        for image_file in os.listdir(class_folder_path):
            if '_83.npz' in image_file:
                self.X_data.append(class_folder_path + image_file)
                self.y_true.append(label_counter)
        label_counter += 1
        
    self.X_data = np.array(self.X_data)
    self.y_true = np.array(self.y_true)

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.y_true) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    #print('Index: {}'.format(index))
    # Generate indexes of the batch
    indexes = self.indexes[index*(self.batch_size//2):(index+1)*(self.batch_size//2)]
#     list_images_temp = [self.images[k] for k in indexes]

    # Generate data
    # X, y = self.__data_generation(list_images_temp)
    X, y = self.__data_generation(indexes)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.y_true))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_idxs):
    '''
    Generate data with the specified batch size
    '''
    # Initialization
    X = np.empty((self.batch_size, 224, 224, 83), dtype=np.uint8)
    y = np.empty((self.batch_size, 42), dtype=np.uint8)

    for i, idx in enumerate(list_idxs):
#       temp = np.load(self.X_data[idx])
#       X[i,] = temp['data']
      temp = cv2.imread(self.X_data[idx])
      temp = cv2.resize(temp, (224, 224))
      temp = np.concatenate([temp, temp], -1)
        
      img_augs = [temp]
      rand_augs = np.random.choice(np.arange(len(self.augs)), 4)
      for idx in rand_augs:
        aug = self.augs[idx]
        img_augs.append(aug(image = temp))
    
      img_augs = np.array(img_augs)
      idx0, idx1 = np.random.choice(np.arange(len(img_augs)), 2)
    
      X[i * 2,] = img_augs[idx0]
      y[i * 2,] = np.zeros((42), np.uint8)
      y[i * 2,][self.y_true[idx]] = 1
        
      X[i * 2 + 1,] = img_augs[idx0]
      y[i * 2 + 1,] = np.zeros((42), np.uint8)
      y[i * 2 + 1,][self.y_true[idx]] = 1

    return X, y