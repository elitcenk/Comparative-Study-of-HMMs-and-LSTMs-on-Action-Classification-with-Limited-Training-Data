__author__ = 'calp'

import matplotlib.pyplot as plt
import os
import itertools
from sklearn import preprocessing
from sklearn.decomposition import PCA
from hmmlearn import hmm
import h5py
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np


class VideoRecognizer:
    def __init__(self, args):
        self.test_person = args.test_person
        if hasattr(args, "left2Right"):
            self.left2Right = args.left2Right
        if hasattr(args, "state_number"):
            self.state_number = args.state_number
        self.experiment = args.experiment
        self.model = dict()
        if not os.path.exists(self.experiment + '/model'):
            os.makedirs(self.experiment + '/model')

        if not os.path.exists(self.experiment + '/model/scaler'):
            os.makedirs(self.experiment + '/model/scaler')
        if not os.path.exists(self.experiment + '/model/pca'):
            os.makedirs(self.experiment + '/model/pca')
        if not os.path.exists(self.experiment + '/model/hmm'):
            os.makedirs(self.experiment + '/model/hmm')

    def testVideos(self):
        videos_categorys = {}
        predicted = []
        expected = []
        output_file = h5py.File(self.experiment + '/features.hdf5', 'r+')
        for video_key in output_file.keys():
            category_name = video_key.split('_')[1]
            if category_name not in videos_categorys:
                videos_categorys[category_name] = []
            videos_categorys[category_name].append(video_key)
        category_names = []
        for category_name in videos_categorys:
            category_names.append(category_name)
            print('Start test category {}'.format(category_name))
            videos_ids = [v for v in videos_categorys[category_name] if v.split('_')[0] in self.test_person and v.split('_')[1] == category_name]
            images = []
            for video_id in videos_ids:
                images.append(output_file[video_id][...])

            print('Starting test {}'.format(category_name))
            for data in images:
                data = joblib.load(self.experiment + '/model/scaler/' + category_name + ".pkl").transform(data)
                data = joblib.load(self.experiment + '/model/pca/' + category_name + ".pkl").transform(data)
                for index in range(data.__len__() - 15):
                    image = data[index: index + 15]
                    max = 0
                    label = category_name
                    for key1 in videos_categorys:
                        score = joblib.load(self.experiment + '/model/hmm/' + key1 + '.pkl').score(image)
                        if score > max:
                            max = score
                            label = key1
                    expected.append(category_name)
                    predicted.append(label)
        f1 = open(self.experiment + '/output.txt', 'w+')
        print("Classification report for classifier \n%s\n" % (metrics.classification_report(expected, predicted)))
        f1.write("Classification report for classifier \n%s\n" % (metrics.classification_report(expected, predicted)))
        cm = metrics.confusion_matrix(expected, predicted)
        print("Confusion matrix:\n%s" % cm)
        f1.write("Confusion matrix:\n%s" % cm)
        category_names.sort()
        self.plotConfusionMatrix(expected, predicted, category_names)

    def trainVideos(self):
        videos_categorys = {}
        output_file = h5py.File(self.experiment + '/features.hdf5', 'r+')
        for video_key in output_file.keys():
            category_name = video_key.split('_')[1]
            if category_name not in videos_categorys:
                videos_categorys[category_name] = []
            videos_categorys[category_name].append(video_key)
        for category_name in videos_categorys:
            print('Start training category {}'.format(category_name))
            videos_ids = [v for v in videos_categorys[category_name] if v.split('_')[0] not in self.test_person and v.split('_')[1] == category_name]
            images = []
            for video_id in videos_ids:
                images.append(output_file[video_id][...])
            images = np.array(images)
            markov_model, std_scale, std_scale_pca = self.train(images)
            joblib.dump(markov_model, self.experiment + "/model/hmm/" + category_name + ".pkl")
            joblib.dump(std_scale, self.experiment + "/model/scaler/" + category_name + ".pkl")
            joblib.dump(std_scale_pca, self.experiment + "/model/pca/" + category_name + ".pkl")

    def train(self, images):
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)
            length.append(file.__len__())
        std_scale = preprocessing.StandardScaler()
        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)
        std_scale_pca = PCA()
        std_scale_pca.fit(scaled_images)
        scaled_images = std_scale_pca.transform(scaled_images)

        ####################TRAIN#########################
        markov_model = hmm.GaussianHMM(n_components=self.state_number, n_iter=1000, random_state=55, transmat_prior=1.00001)
        if self.left2Right:
            startprob, transmat = self.initByBakis(self.state_number, 2)
            markov_model.init_params = "c"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat
        markov_model.fit(scaled_images, length)
        return markov_model, std_scale, std_scale_pca

    def plotConfusionMatrix(self, expected, predicted, target_names):
        cm = metrics.confusion_matrix(expected, predicted)
        np.set_printoptions(precision=2)
        # print("Confusion matrix:\n%s" % cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "%.2f" % round(cm[i, j], 2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig(self.experiment + '/output.png')

    def initByBakis(self, nComp, bakisLevel):
        ''' init start_prob and transmat_prob by Bakis model '''
        startprobPrior = np.zeros(nComp)
        startprobPrior[0: bakisLevel - 1] = 1. / (bakisLevel - 1)
        transmatPrior = self.getTransmatPrior(nComp, bakisLevel)
        return startprobPrior, transmatPrior

    def getTransmatPrior(self, nComp, bakisLevel):
        ''' get transmat prior '''
        transmatPrior = (1. / bakisLevel) * np.eye(nComp)
        for i in range(nComp - (bakisLevel - 1)):
            for j in range(bakisLevel - 1):
                transmatPrior[i, i + j + 1] = 1. / bakisLevel

        for i in range(nComp - bakisLevel + 1, nComp):
            for j in range(nComp - i - j):
                transmatPrior[i, i + j] = 1. / (nComp - i)

        return transmatPrior
