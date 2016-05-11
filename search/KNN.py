import numpy as np
import time

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from utils.fio import get_absolute_path, parse_feature_map
from utils.fio import get_config
from utils.transcription import get_transcription, get_word, WordCoord


class KNN:
    def __init__(self):
        self._k = 1
        self._d = None
        self._train = None
        self._valid = None

    def fit(self, dataset):
        if type(dataset) != DataSet:
            raise IOError("The input has to be a KNN.DataSet")

        self._train = dataset

    def parse(self, items=None, id_filter=None):
        print('parse features')
        config = get_config()
        fmp = get_absolute_path(config.get('KWS.features', 'file'))
        ids, imgs, mats = parse_feature_map(fmp, items=items, id_filter=id_filter)

        print('parse transcription')
        trans = get_transcription()
        words = []
        coords = []
        for id in ids:
            word = get_word(id, data=trans)
            words.append(str(word))
            coords.append(WordCoord(id))

        self._train = DataSet(ids, imgs, mats, np.array(coords), np.array(words))

    def parse_train_and_valid(self):
        self.parse()

        # get the training subset
        config = get_config()
        tp = get_absolute_path(config.get('KWS', 'testing'))
        dids = []
        for line in open(tp, 'r'):
            did = line.strip()
            if len(did) == 3:
                dids.append(did)

        # create the index for the features
        index = np.array([dids.count(x.doc_id) == 1 for x in self._train.coords], dtype=bool)

        # Put the data in memory
        self._valid = DataSet(self._train.Y[index],
                              self._train.imgs[index],
                              self._train.X[index],
                              self._train.words[index],
                              self._train.coords[index])
        index = ~index  # (tested if the valid doc ids are indeed the complement)
        self._train = DataSet(self._train.Y[index],
                              self._train.imgs[index],
                              self._train.X[index],
                              self._train.words[index],
                              self._train.coords[index])

    def set_k(self, value):
        self._k = value

    def training_score(self):
        d = PairWiseDist(self._train.X)
        correct = 0
        for i in range(self._train.N):
            dis, ind = d.get_dists(i)
            word = self.vote(dis, self._train.words[ind])
            if word == self._train.words[i]:
                correct += 1
            else:
                print('misclassified "%s" as "%s"' % (self._train.words[i], word))

        return correct / self._train.N

    def vote(self, dists, lbls):
        t = [(x, y) for x, y in zip(dists, lbls)]
        t = sorted(t, key=lambda x: x[0])
        # take n nearest
        kn = t[0:self._k]

        # the dictator case
        if self._k == 1:
            return kn[1]

        # count the labels
        votes = [x[1] for x in kn]
        candidates = [x for x in set(votes)]
        counts = [votes.count(x) for x in candidates]

        if len(counts) == len(set(counts)):
            index = np.argmax(counts)
            return candidates[index]
        else:
            return kn[0][1]  # tie break the tie with the minimum distance

    def classify(self, mat, img, tol=5):
        # subset index
        h_min = img[0] - tol
        h_max = img[0] + tol
        w_min = img[1] - tol
        w_max = img[1] + tol

        i = ((self._train.h >= h_min) & (h_max <= self._train.h)) & \
            (self._train.w >= w_min) & (w_max <= self._train.h)

        x = np.append(mat, self._train.X[i])
        pd = PairWiseDist(x)
        d, j = pd.get_dists(0)
        y = self.vote(d, self._train.words[j])

        return y


class PairWiseDist:
    def __init__(self, x):
        self._M = x.shape[0]

        t1 = time.time()
        n = int(((self._M * self._M - self._M) / 2))
        print('compute distances (%i)' % n)
        rows = np.zeros((n,), dtype=int)
        cols = np.zeros((n,), dtype=int)
        dist = np.zeros((n,))
        c = 0
        for i in range(self._M - 1):
            for j in range(i + 1, self._M):
                print('%i, %i' % (i, j))
                rows[c] = i
                cols[c] = j
                dist[c], _ = fastdtw(x[i], x[j], dist=euclidean)
                c += 1
                if c % 1000 == 0:
                    print('.', end='', flush=True)

        et = time.time() - t1
        print('\n%i pairs in %s sec.' % (n, et))
        self._i = rows
        self._j = cols
        self._d = dist

    def get_dist(self, i, j):
        """
        return the distance between the items with index i and j.
        """
        index = (self._i == i) & (self._j == j)
        return self._d[index]

    def get_dists(self, i):
        """
        return all the distances between item i and the all the other
        items. Additionally the index of the other items are returned.
        """
        a = self._i == i
        b = self._j == i
        ab = a | b

        j = np.zeros((self._i.shape[0], 2), dtype=int)
        j[a, 0] = self._j[a]
        j[b, 1] = self._i[b]
        j = j.max(axis=1)
        j = j[np.where(ab)]

        d = self._d[ab]

        return d, j


class DataSet:
    def __init__(self, ids, imgs, mats, coords, words):
        self.Y = ids
        self.imgs = imgs
        self.h = np.array([x[0] for x in imgs])
        self.w = np.array([x[1] for x in imgs])
        self.rank = np.array([x[2] for x in imgs])
        self.X = mats
        self.coords = coords
        self.words = words
        self.N = len(ids)
