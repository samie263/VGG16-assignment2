from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from glob import glob
import numpy as np
import json
import cv2
import os

VGG16_Model = load_model('vgg16Model.h5')


class ObjectDetection(object):

    def __init__(self):
        self._objects = []

    def video_to_frames(self, video):
        print('splitting to frames...')
        frame_name = 'static/frames/frame'
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        count = 0
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if success:
                cv2.imwrite(frame_name + str(count) + '.jpg', frame)
                print(frame)
            else:
                break
            count = count + 1
        vidcap.release()
        cv2.destroyAllWindows()
        print('Done splitting')

    def detect(self):
        print('feeding frames to vgg16...')
        for frame in self.get_frames():
            image = load_img(frame, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            y_pred = VGG16_Model.predict(image)
            label = decode_predictions(y_pred)
            self._objects.append(label[0][1][1])
        print('Done feeding')
        with open('detected_objects.txt', 'w') as f:
            f.write(json.dumps(self._objects))

    def get_frames(self):
        frames_arr = glob("static/frames/*.jpg")
        return frames_arr

    def get_objects(self):
        return self._objects

    def search_objects(self, _object):
        print('searching...')
        with open('detected_objects.txt', 'r') as objects_file:
            objects = list(json.loads(objects_file.read()))
        search_results = []
        if _object in set(objects):
            for index in range(len(objects)):
                if _object.__eq__(objects[index]):
                    img_url = self.get_frames()[index]
                    img_url = os.path.join('frames/', img_url.split('\\')[1])
                    search_results.append(img_url)
        else:
            return 'Object does not found'
        return search_results

    def read_objects(self):
        with open('detected_objects.txt', 'r') as objects_file:
            objects = set(json.loads(objects_file.read()))
        return objects
