import sys
import cv2
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
#from utils.inference import load_image
from utils.preprocessor import preprocess_input

def _load_fer2013(dataset_path,image_size):
    data = pd.read_csv(dataset_path)
    #here is the change 
    data = data.loc[ data['Usage'] == 'PublicTest']
    #data = data.loc[data['Usage']=='PrivateTest']
    pixels = data['pixels'].tolist()
    #width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
       face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    	#face = np.asarray(face).reshape(width, height)
       face = np.asarray(face)
       face = cv2.resize(face.astype('uint8'), image_size)
       faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    #faces = np.expand_dims(faces, 0)
    #faces = np.expand_dims(faces, -1)
    #emotions = pd.get_dummies(data['emotion']).as_matrix()
    emotions = data['emotion']
    return faces, emotions

# parameters for loading data and images
image_path = 'fer2013.csv'

detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'fer2013_mini_XCEPTION.56-0.65.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
#emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

#list to store labels
pred_labels =[]
#count = 0
#image_size = (64,64)
faces, emotions = _load_fer2013(image_path,image_size)
for gray_face in faces:
  gray_face = np.expand_dims(gray_face, 0)
  gray_face = np.expand_dims(gray_face, -1)
  emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
  pred_labels.append(emotion_label_arg)

		
pred_labels = np.asarray(pred_labels)
#print(pred_labels.shape)
print(pred_labels[:20])
print(emotions[:20])