import os
import webbrowser

from datetime import datetime
import numpy as np
import time

from scipy.spatial.distance import euclidean

from dtwextension import dtwdistance

from search.visualize import display_all_occurences
from utils.fio import get_absolute_path, parse_feature_map
from utils.fio import get_config
from utils.transcription import get_transcription, get_word, WordCoord


class KWS:
    def __init__(self):
        config = get_config()
        self._k = int(config.get('KWS.classifier', 'k'))
        self._tol_v = int(config.get('KWS.classifier', 'tol_ver'))
        self._tol_h = int(config.get('KWS.classifier', 'tol_hor'))
        self._d = None
        self.train = None
        self.valid = None
        self.data = None
        self.index = None
        self._log = None

    def fit(self, dataset):
        """
        Set the data used for knn classification
        """
        if type(dataset) != DataSet:
            raise IOError("The input has to be a KNN.DataSet")

        self.train = dataset

        if (self.train is None) or self.train.N <= 1:
            raise IOError("There is no data. Define dataset input or use KNN.parse().")

    def set_data(self, dataset):
        """
        Set the dataset for indexing and searching
        """
        if type(dataset) != DataSet:
            raise IOError("The input has to be a KNN.DataSet")

        self.data = dataset

    def load_train_and_valid(self):
        """
        Convenience method to load the training and validation data accoring
        to the specifications in the config.ini
        """
        dataset = DataSet.parse()
        # self.parse()

        # get the training subset
        config = get_config()
        tp = get_absolute_path(config.get('KWS', 'testing'))
        dids = []
        for line in open(tp, 'r'):
            did = line.strip()
            if len(did) == 3:
                dids.append(did)

        # create the index for the features
        index = np.array([dids.count(x.doc_id) == 1 for x in dataset.coords], dtype=bool)

        # Put the data in memory
        self.valid = DataSet(dataset.Y[index],
                             dataset.imgs[index],
                             dataset.X[index],
                             dataset.coords[index])
        index = ~index  # (tested if the valid doc ids are indeed the complement)
        self.train = DataSet(dataset.Y[index],
                             dataset.imgs[index],
                             dataset.X[index],
                             dataset.coords[index])

    def set_k(self, value):
        self._k = value

    def set_tol(self, ver, hor):
        self._tol_h = hor
        self._tol_v = ver

    def training_score(self):
        self.create_log()

        d = PairWiseDist(self.train.X)
        correct = 0.0
        for i in range(self.train.N):
            dis, ind = d.get_dists(i)
            lbl = self.train.Y[ind]
            word, md, cnt = self.vote(dis, lbl)
            gt = self.train.Y[i]
            ns = sum(self.train.Y == word)

            if self.clean_word(word) == self.clean_word(gt):
                correct += 1
                msg = ''
            else:
                msg = 'x   '

            msg += '%s -> %s' % (gt, word)
            msg += ' ' * (40 - len(msg))
            msg += '#train-words: %i\t\tmin-dist: %.0f\t\tvotes: %i\tid: %s\n' \
                   % (ns, md, cnt, self.train.coords[i].id)

            self.log(msg)

        acc = correct / float(self.train.N)
        msg = '\nAccuracy: %.2f (%i samples)' % (acc, self.train.N)
        self.log(msg)

        return acc

    @staticmethod
    def clean_word(s):
        return s.replace(',', '').replace('.', '').replace(';', '').replace(':','')

    def vote(self, dists, lbls):
        t = [(x, y) for x, y in zip(dists, lbls)]
        t = sorted(t, key=lambda x: x[0])

        # take n nearest
        kn = t[0: self._k - 1]

        # the dictator case
        if self._k == 1:
            return kn[1], kn[0], 0

        # count the labels
        votes = [x[1] for x in kn]
        candidates = [x for x in set(votes)]
        counts = [votes.count(x) for x in candidates]

        # among the candidates with the most votes select the nearest
        max_count = max(counts)
        min_dists = []
        best_cans = []
        max_counts = []
        for can, cnt in zip(candidates, counts):
            if cnt == max_count:
                dists = []
                for x in t:
                    if x[1] == can:
                        dists.append(x[0])

                min_dists.append(min(dists))
                max_counts.append(cnt)
                best_cans.append(can)

        index = np.argmin(min_dists)
        return best_cans[index], min_dists[index], max_counts[index]

    def classify(self, mat, coord, img=None):
        """
        classify a single sample (mat).
        mat KxN is a feature matrix from an image. N features computed for K windows.
        If the tuple img (image height, image width, rank) is defined,
        the distance computation will be constrained by the image parameter.
        """
        t0 = time.time()

        config = get_config()
        prune = config.get('KWS.classifier', 'prune') == 'True'

        if img is None or not prune:
            x = self.train.X
            y = self.train.Y
            c = np.array([x.id for x in self.train.coords])
        else:
            # subset index
            h_min = img[0] - self._tol_v
            h_max = img[0] + self._tol_v
            w_min = img[1] - self._tol_h
            w_max = img[1] + self._tol_h

            i = ((self.train.h >= h_min) & (h_max >= self.train.h)) & \
                (self.train.w >= w_min) & (w_max >= self.train.h) & \
                (self.train.rank == img[2])
            # x = np.append([mat], self.train.X[i])
            x = self.train.X[i]
            y = self.train.Y[i]
            c = np.array([x.id for x in self.train.coords[i]])

        # default output ...
        lbl = '???'
        md = -1
        cnt = -1
        nc = x.shape[0]

        # ... because there are a lot of ways to fail
        if nc > 0:
            not_itself = coord != c
            # print(sum(~not_itself))
            x = x[not_itself]
            y = y[not_itself]
            nc = x.shape[0]
            if nc > 0:
                d = np.zeros((nc,))
                for i, m in enumerate(x):
                    # d[i], _ = fastdtw(mat, m, dist=euclidean)
                    d[i] = dtwdistance(mat, m)

                lbl, md, cnt = self.vote(d, y)

        return lbl, nc, md, cnt, time.time() - t0

    def create_log(self):
        config = get_config()
        cp = get_absolute_path(config.get('KWS.classifier', 'file'))
        cd = os.path.dirname(cp)
        if not os.path.exists(cd):
            os.mkdir(cd)

        self._log = os.path.join(cd, datetime.now().strftime('%y-%m-%d_%H-%M_') + os.path.basename(cp))
        msg = 'Testing\nk=%i\nvertical tolerance=%i\nhorizontal tolerance=%i\n# training samples: %i\n' % \
              (self._k, self._tol_v, self._tol_h, self.train.N)
        print(msg, end='')
        f = open(self._log, 'w+')
        f.write(msg)

    def log(self, msg):
        print(msg, end='', flush=True)
        f = open(self._log, 'a')
        f.write(msg)
        f.close()

    def test(self, mats, lbls, coords, imgs=None):
        """
        Create a test log given some data
        """
        self.create_log()
        self.log('# validation samples: %i\n' % len(mats))

        img = None
        cor = 0.0
        ntr = 0.0
        for n, mat in enumerate(mats):
            if imgs is not None:
                img = imgs[n]

            coord = coords[n].id
            lbl = lbls[n]
            ns = sum(self.train.Y == lbl)
            ntr += ns == 0

            y, nc, md, cnt, t = self.classify(mat, coord, img=img)

            if self.clean_word(y) != self.clean_word(lbl):
                msg = 'x   '
            else:
                cor += 1
                msg = ''

            msg += '%s -> %s' % (lbl, y)
            msg += ' ' * (47 - len(msg))
            msg += self.append_str('#train-words', ns)
            msg += self.append_str('#candidates', nc)
            msg += self.append_str('min-dist', md)
            msg += self.append_str('votes', cnt)
            msg += self.append_str('cpu-time', t)
            msg += 'id: %s\n' % coord
            self.log(msg)

        acc = cor / float(len(mats))
        msg = "\nOverall Accuracy: %.2f (%i samples)\n" % (acc, len(mats))
        self.log(msg)
        n = float(len(mats) - ntr)
        acc = cor / n
        msg = "Accuracy (without counting new samples with no training data): %.2f (%i samples)\n" % (acc, n)
        self.log(msg)

    @staticmethod
    def append_str(txt, val):
        if type(val) == float:
            msg = '%s: %0.2f' % (txt, val)
        elif type(val) == str or type(val) == np.str_:
            msg = '%s: %s' % (txt, val)
        else:
            msg = '%s: %i' % (txt, val)
        msg += ' ' * (8 - len(msg) + len(txt))

        return msg

    def create_index(self, data=None):
        """
        create a word - document-coordinate index given some (new) data
        """
        t0 = time.time()

        if data is not None:
            self.data = data

        print('indexing ', end='')

        self.index = {}
        img = None
        for n, mat in enumerate(self.data.X):
            if self.data.imgs is not None:
                img = self.data.imgs[n]

            coord = self.data.coords[n].id

            if n % 10 == 0:
                print('.', end='', flush=True)

            lbl, nc, md, cnt, t = self.classify(mat, coord, img=img)
            if lbl in self.index:
                self.index[lbl].append(coord)
            else:
                self.index[lbl] = [coord]

        print('')
        twc = 0
        for key in self.index:
            wc = len(self.index[key])
            lc = len(str(wc))
            lw = len(key)
            msg = '\t' + key + ' ' * (25 - lc - lw) + str(wc) + '-times'
            print(msg)
            twc += wc

        print('indexed %i words in %.2f sec' % (twc, time.time() - t0))

    def search_word(self, word):
        """
        Search for a word and display the results as html in default webbrowser
        """
        if self.index is None:
            self.create_index()

        if word not in self.index:
            print('%s is not found' % word)
            return

        d = get_absolute_path('search/vis/')
        if not os.path.exists(d):
            os.mkdir(d)

        outputfile = os.path.join(d, word + '.html')
        display_all_occurences(word, self.index[word], output=outputfile)

        print('searched "%s", found %i occurrences' % (word, len(self.index[word])))
        webbrowser.open('file://' + outputfile)

    def search_pos(self, coord):
        if [str, np.str, np.str_].count(type(coord)) > 0:
            coord = WordCoord(coord)

        for key in self.index:
            if self.index[key].count(coord) > 0:
                self.search_word(key)
                return

            print('word coordinate %s is not indexed' % coord)


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
                # print('%i, %i' % (i, j))
                rows[c] = i
                cols[c] = j
                # dist[c], _ = fastdtw(x[i], x[j], dist=euclidean)
                dist[c] = dtwdistance(x[i], x[j])
                c += 1
                if c % 1000 == 0:
                    print('.', end='', flush=True)

        et = time.time() - t1
        print('%i pairs in %s sec.' % (n, et))
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
    def __init__(self, words, imgs, mats, coords):
        self.Y = words
        self.imgs = imgs
        self.h = np.array([x[0] for x in imgs])
        self.w = np.array([x[1] for x in imgs])
        self.rank = np.array([x[2] for x in imgs])
        self.X = mats
        self.coords = coords
        self.N = len(words)

    def subset(self, index):
        return DataSet(self.X[index], self.imgs[index], self.X[index], self.coords[index])

    @staticmethod
    def parse(filename=None, items=None, id_filter=None):
        print('parse features')
        config = get_config()

        if filename is None:
            fmp = get_absolute_path(config.get('KWS.features', 'file'))
        else:
            fmp = get_absolute_path(filename)

        ids, imgs, mats = parse_feature_map(fmp, items=items, id_filter=id_filter)

        print('parse transcription')
        trans = get_transcription()
        words = []
        coords = []
        for coord in ids:
            word = get_word(coord, data=trans)
            words.append(str(word))
            coords.append(WordCoord(coord))

        return DataSet(np.array(words), imgs, mats, np.array(coords))
