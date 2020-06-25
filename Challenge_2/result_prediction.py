import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

def perform_prediction(images, test_model):
    images = np.array(images)
    predictions = test_model.predict(images)
    predictions = [np.argmax(p) for p in predictions]
    print(predictions)
    return predictions

def main():
    IMAGE_FOLDER = './shopee-product-detection-dataset/test/test/'
    MODEL_WEIGHT_PATH = './Checkpoints/model_and_weights.h5'
    EXPORT_FILE = './little_fat_boys.csv'

    images = []
    files = []
    all_predictions = []

    #Load predicting model
    test_model = load_model(MODEL_WEIGHT_PATH)

    counter = 0
    images = []
    #Get all images
    try:
        for image_file in os.listdir(IMAGE_FOLDER):
            if len(image_file) != 36:
                continue
            if counter == 16:
                predictions = perform_prediction(images, test_model)
                all_predictions += predictions
                images = []
                counter = 0
                
            image = cv2.imread(IMAGE_FOLDER + image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            files.append(image_file)
            counter += 1
        
        if len(images) > 0:
            predictions = perform_prediction(images, test_model)
            all_predictions += predictions
    except KeyboardInterrupt:
        print('Keyboard interrupt')
        pass

    #Save to csv
    print(len(all_predictions))
    df = pd.DataFrame(zip(files, all_predictions), columns=['Filename', 'Category'])
    df.to_csv(EXPORT_FILE, index = False)

if __name__ == '__main__':
    main()