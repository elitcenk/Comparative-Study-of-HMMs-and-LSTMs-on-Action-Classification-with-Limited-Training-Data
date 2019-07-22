__author__ = 'calp'

import argparse

import hmm_util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train human action with HMM')
    parser.add_argument('-s', '--state-number', type=int, dest='state_number', default=5, help='Number of states in the model. (default: %(default)s)')
    parser.add_argument('-tp', '--test-person', type=str, dest='test_person', default=['person02', 'person03', 'person05', 'person06', 'person07', 'person08', 'person09', 'person10', 'person22'], nargs='+', help='Test person. (default: %(default)s)')
    parser.add_argument('-l2r', '--left-2-right', action='store_true',dest='left2Right', default=False, help='Left to right HMM model. (default: %(default)s)')
    parser.add_argument('-exp', '--experiment-name', type=str, dest='experiment', default='Deney1', help='Experiment name. (default: %(default)s)')

    args = parser.parse_args()
    videoRecognizer = hmm_util.VideoRecognizer(args)
    videoRecognizer.trainVideos()
