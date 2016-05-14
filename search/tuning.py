import numpy as np
import matplotlib.pyplot as plt

from ip.preprocess import create_word_mask
from search.KNN import KNN
from utils.fio import get_image_roi
from utils.transcription import get_transcription


def compute_central_heights():
    trc = get_transcription()
    wh = []
    for coord, word in trc:
        roi = get_image_roi(coord)
        msk = create_word_mask(roi)
        img = np.copy(roi)
        img = img.max() - img
        img[msk < 1] = 0
        y = img.sum(1)
        y_max = y.max()
        y[y < y_max * 0.6] = 0
        nz = y.nonzero()
        wh.append(nz[0].shape[0])

        return np.array(wh)


def compute_word_dimensions():
    knn = KNN()
    knn.parse()

    stats = dict()
    for word, img in zip(knn.train.Y, knn.train.imgs):
        if word in stats:
            stats[word].append(img)
        else:
            stats[word] = [img]

    wr = []
    hr = []
    rr = []
    for word in stats:
        w = [x[0] for x in stats[word]]
        h = [x[1] for x in stats[word]]
        r = [x[2] for x in stats[word]]

        wr.append(max(w) - min(w))
        hr.append(max(h) - min(h))
        rr.append(max(r) - min(r))

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.hist(wr, 100)
    plt.xlabel('width ranges')
    ax2 = fig.add_subplot(132)
    ax2.hist(hr, 50)
    plt.xlabel('height ranges')
    ax3 = fig.add_subplot(133)
    ax3.hist(rr)
    plt.show()


if __name__ == '__main__':
    compute_word_dimensions()
