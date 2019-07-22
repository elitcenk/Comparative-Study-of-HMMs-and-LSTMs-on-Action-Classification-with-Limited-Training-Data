__author__ = 'calp'

import argparse
import os
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.layers import Dense, Flatten
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.models import load_model

import cv2


def extractFrames(video_path, target, resize=None):
    images_x = []
    images_y = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        if resize:
            # The resize of CV2 requires pass firts width and then height
            frame = cv2.resize(frame, (resize[1], resize[0]))
        images_x.append(frame)
        images_y.append(target)

    video = np.array(images_x, dtype=np.float32)
    return video, images_y


def loadVideos(videos_dir, sequence_file):
    images = []
    labels = []
    classes = {'boxing': 0, 'handclapping': 1, 'handwaving': 2, 'jogging': 3, 'running': 4, 'walking': 5}
    file = open(sequence_file, "r")
    for line in file:
        if line != '\n':
            splitted = line.split("frames")
            frames, label = extractFrames(os.path.join(videos_dir, splitted[0].strip()), classes[splitted[0].strip().split('_')[1]])
            for startstop in splitted[1].split(', '):
                startstopsp = startstop.split('-')
                images.extend(frames[int(startstopsp[0]) - 1:int(startstopsp[1]) - 1])
                labels.extend(label[int(startstopsp[0]) - 1:int(startstopsp[1]) - 1])
        else:
            images = preprocess_input(np.array(images))
            labels = np_utils.to_categorical(labels, 6)
            yield images, labels
            images = []
            labels = []


def trainVideos(batch_size, videos_dir,sequence_file, loaded_model):
    if loaded_model is not None:
        model = load_model(loaded_model)
        print("Model loaded")
    else:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(120, 160, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.get_layer('block5_pool').output
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dense(6, activation='softmax', name='predictions')(x)
        model = Model(input=base_model.input, output=x)
        # model.load_weights("D:\\Apps\\action_recognition\\deep\\data\\models\\kthweights.hdf5")

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=False )
    for images, labels in loadVideos(videos_dir,sequence_file):
        model.fit(images, labels, batch_size=batch_size, epochs=1, validation_split=0.0, callbacks=[checkpointer])
    model.save_weights('savedWeights.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video using CNN')
    parser.add_argument('-d', '--videos-dir', type=str, dest='directory', default='D:/Apps/videos/kth', help='videos directory (default: %(default)s)')
    parser.add_argument('-l', '--loaded-model', type=str, dest='loaded_model', help='loaded model path (default: %(default)s)')
    parser.add_argument('-s', '--sequence-file', type=str, dest='sequences', default='D:/Apps/action_recognition/deep/data/00sequences.txt', help='Video sequences text. (default: %(default)s)')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=32, help='batch size when extracting features (default: %(default)s)')
    args = parser.parse_args()

    trainVideos(args.batch_size, args.directory, args.sequences,args.loaded_model)
