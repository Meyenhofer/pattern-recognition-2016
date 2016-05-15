import numpy as np
import matplotlib.pyplot as plt

from ip.preprocess import create_word_mask
from search.KNN import KNN
from utils.fio import get_image_roi, get_config
from utils.transcription import get_transcription


def compute_central_heights():
    config = get_config()
    rh = float(config.get('KWS.prepro', 'relative_height'))
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
        y[y < y_max * rh] = 0
        nz = y.nonzero()
        wh.append(nz[0].shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(wh, 30, normed=True)
    plt.xlabel('central peak heights of vertical word projection')
    plt.show()

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
        h = [x[0] for x in stats[word]]
        w = [x[1] for x in stats[word]]
        r = [x[2] for x in stats[word]]

        wr.append(max(w) - min(w))
        hr.append(max(h) - min(h))
        rr.append(max(r) - min(r))

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.hist(wr, 30, normed=1, histtype='step', cumulative=1)
    ax1.margins(x=0)
    plt.xlabel('word width ranges')
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor')
    plt.minorticks_on()
    ax2 = fig.add_subplot(132)
    ax2.hist(hr, 30, normed=1, histtype='step', cumulative=1)
    ax2.margins(x=0)
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor')
    plt.minorticks_on()
    plt.xlabel('word height ranges')
    plt.title('cumulative distributions')
    ax3 = fig.add_subplot(133)
    ax3.hist(rr, normed=1, histtype='step', cumulative=1)
    ax3.margins(x=0)
    plt.xlabel('word rank range')
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor')
    plt.minorticks_on()
    plt.show()


if __name__ == '__main__':
    """
    Produce some plots to help tuning the parameter of the key word search.
    """
    compute_word_dimensions()

    ch = compute_central_heights()
