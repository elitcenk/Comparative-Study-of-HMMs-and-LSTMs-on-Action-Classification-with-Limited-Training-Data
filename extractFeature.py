__author__ = 'calp'
import argparse
import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
import h5py

import cv2


def video_to_array(video_path, resize=None, start_frame=0, end_frame=None, length=None, dim_ordering='th', feature_type=None, mhi=False):
    """ Convert the video at the path given in to an array
    Args:
        video_path (string): path where the video is stored
        resize (Optional[tupple(int)]): desired size for the output video.
            Dimensions are: height, width
        start_frame (Optional[int]): Number of the frame to start to read
            the video
        end_frame (Optional[int]): Number of the frame to end reading the
            video.
        length (Optional[int]): Number of frames of length you want to read
            the video from the start_frame. This override the end_frame
            given before.
    Returns:
        video (nparray): Array with all the data corresponding to the video
                         given. Order of dimensions are: channels, length
                         (temporal), height, width.
    Raises:
        Exception: If the video could not be opened
    """
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

    if dim_ordering not in ('th', 'tf'):
        raise Exception('Invalid dim_ordering')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video')

    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    if start_frame >= num_frames or start_frame < 0:
        raise Exception('Invalid initial frame given')
    # Set up the initial frame to start reading
    cap.set(CAP_PROP_POS_FRAMES, start_frame)
    # Set up until which frame to read
    if end_frame:
        end_frame = end_frame if end_frame < num_frames else num_frames
    elif length:
        end_frame = start_frame + length
        end_frame = end_frame if end_frame < num_frames else num_frames
    else:
        end_frame = num_frames
    if end_frame < start_frame:
        raise Exception('Invalid ending position')

    frames = []
    previous = None
    mhiFrame = None
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
        if resize:
            # The resize of CV2 requires pass first width and then height
            frame = cv2.resize(frame, (resize[1], resize[0]))
        if mhi:
            if previous is not None:
                silhouette = cv2.addWeighted(previous, -1.0, frame, 1.0, 0)
                mhiFrame = cv2.addWeighted(silhouette, 1.0, mhiFrame, 0.9, 0)
            else:
                mhiFrame = np.zeros(frame.shape, frame.dtype)
            previous = frame.copy()

            # cv2.imshow('MHI', mhiFrame)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
            frame = mhiFrame.copy()
        if feature_type == 'Hu':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.HuMoments(cv2.moments(frame)).flatten()
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    if dim_ordering == 'th':
        video = video.transpose(3, 0, 1, 2)
    return video


def extract_features(videos_dir, weights_path, sequence_file, feature_type, experiment, mhi=False, resize=None):
    if not os.path.exists(experiment):
        os.makedirs(experiment)
    output_path = os.path.join(experiment + '/features.hdf5')
    mode = 'r+' if os.path.exists(output_path) else 'w'
    # Extract the ids of the videos already extracted its features
    output_file = h5py.File(output_path, mode)
    extracted_videos = output_file.keys()
    liste = {}
    for extracted_video in extracted_videos:
        if extracted_video[:-1] in liste:
            liste[extracted_video[:-1]] += 1
        else:
            liste[extracted_video[:-1]] = 1
    for v in liste:
        if liste[v] != 4:
            print(v)
    videos_ids = [v[:-4] for v in os.listdir(videos_dir) if v[-4:] == '.avi']

    # Lets remove from the list videos_ids, the ones already extracted its features
    videos_ids_to_extract = list(set(videos_ids) - set(liste))
    file = open(sequence_file, "r")
    sequences = {}
    for line in file:
        if line != '\n':
            splitted = line.split("frames")
            sequences[splitted[0].strip()] = splitted[1].strip()

    nb_videos = len(videos_ids_to_extract)
    print('Total number of videos: {}'.format(len(videos_ids)))
    print('Videos already extracted its features: {}'.format(len(liste)))
    print('Videos to extract its features: {}'.format(nb_videos))
    output_file.close()
    model = None
    if feature_type == 'CNN':
        print('Loading model')
        base_model = VGG16(weights=None, include_top=False, input_shape=(120, 160, 3))
        x = base_model.get_layer('block5_pool').output
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dense(6, activation='softmax', name='predictions')(x)
        model = Model(input=base_model.input, output=x)
        model.load_weights(weights_path)

        print('Compiling model')
        model.compile(optimizer='sgd', loss='mse')
        print('Compiling done!')
    print('Starting extracting features')

    for video_id in videos_ids_to_extract:
        path = videos_dir + '\\' + video_id + '.avi'
        print('Start extracting features from video {}'.format(video_id))
        vid_array = video_to_array(path, resize=resize, start_frame=0, dim_ordering="tf", mhi=mhi, feature_type=feature_type)
        counter = 1
        for startstop in sequences[video_id + '.avi'].split(','):
            startstopsp = startstop.strip().split('-')
            features = vid_array[int(startstopsp[0]) - 1:int(startstopsp[1]) - 1]
            if model is not None:
                features = model.predict(features, batch_size=32)
            print('Extracted features from video {}'.format(video_id))
            with h5py.File(output_path, 'r+') as f:
                f.create_dataset(video_id + str(counter), data=features, dtype='float32')
            counter += 1

        print('Saved video {}'.format(video_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features')
    parser.add_argument('-r', '--resize', type=int, dest='resize', default=None, help='Frame resize ratio. (default: %(default)s)')
    parser.add_argument('-mhi', '--mhi', dest='mhi', default=False, action='store_true', help='Do use MHI Feature extraction? (default: %(default)s)')
    parser.add_argument('-f', '--feature-type', type=str, dest='feature_type', default='Hu', help='Feature type. *"Hu" *"CNN"  (default: %(default)s)')
    parser.add_argument('-d', '--videos-dir', type=str, dest='directory', default='D:\Apps\\videos\\kth', help='videos directory (default: %(default)s)')
    parser.add_argument('-s', '--sequence-file', type=str, dest='sequences', default='D:/Apps/action_recognition/deep/data/00sequences.txt', help='Video sequences text. (default: %(default)s)')
    parser.add_argument('-w', '--weights-dir', type=str, dest='weights', default='D:\Apps\\action_recognition\\deep\\data\\models\\kthweights.hdf5', help='model weights path(default: %(default)s)')
    parser.add_argument('-exp', '--experiment-name', type=str, dest='experiment', default='Deney1', help='Experiment name. (default: %(default)s)')

    args = parser.parse_args()

    extract_features(args.directory, args.weights, args.sequences, args.feature_type, args.experiment, args.mhi, args.resize)
