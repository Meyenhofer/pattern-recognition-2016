import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import moment
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def compute_features(roi, window_width=1, step_size=3, blocks=3):
    w = roi.shape[1]
    msk = roi > 0
    # ske = skeletonize(np.array(msk, dtype=bool))

    f = []

    x1 = 0
    x2 = x1 + window_width
    while True:
        # get pixel subset
        bw = msk[:, x1:x2]
        gs = roi[:, x1:x2]
        gsm = gs[bw]
        if len(gsm) == 0:
            gsm = np.array([0])

        h = len(bw)

        # gray-scale features
        fv = [np.mean(gsm),
              moment(gsm, moment=2),
              moment(gsm, moment=3),
              moment(gsm, moment=4)]

        # binary features over the entire window
        fv.extend(binary_features(bw))
        # fv.extend(binary_features(ske))

        # binary features over blocks of the window
        if blocks > 1:
            bh = int(np.floor(h / blocks))
            for a, b in zip(range(0, blocks * bh, bh), range(bh, h + 1, bh)):
                fv.extend(bw[a:b])

        f.append(fv)

        x1 = x2 + step_size
        x2 = x1 + window_width
        if x2 > w:
            break

    return f


def binary_features(bw):
    h = bw.shape[0]
    gra = bw[0:-1] - bw[1:]

    # number of black-white and white-black transitions
    bwt, wbt = transitions(gra)
    # pixels not on the contour
    dk = (gra == 0) & (bw[0:-1] > 0)
    dk = dk.sum() / h
    # foreground fraction
    fgf = bw.sum() / h

    mi = np.where(bw)[0]
    if len(mi) == 0:
        top = 0
        bot = 0
        cen = 0
    else:
        # top
        top = np.min(mi) / h
        # bottom
        bot = np.max(mi) / h
        # average mask positions (relative to height)
        cen = np.median(mi) / h

    return [bwt, wbt, dk, fgf, top, bot, cen]


def transitions(bin_img):
    """
    Returns the number of black to white and white to black transitions.
    The two numbers will only be different when the image does not end in the
    same colour as it started.
    """
    black_white_count = 0
    white_black_count = 0
    current_state = bin_img[0][0]
    for lis in bin_img:
        for val in lis:
            if val is not current_state:
                current_state = val
                if val:
                    black_white_count += 1
                else:
                    white_black_count += 1

    return black_white_count, white_black_count


def word_symmetry(img, rel_height=0.5, ppw=20, spw=8, show=False):
    """
    returns [1...4]
    0: lowercase like a, e, i, o, u
    1: letters like b, t, h,
    2: letters like g, p, q
    3: letters like f
    """
    y = img.sum(1)
    ym = y.max()
    ya = y.argmax()
    arm = argrelextrema(y, np.greater)[0]
    rmv = y[arm]
    ch = (ym * rel_height)
    i = y > ch
    ii = np.where(i)[0]

    u = np.where(np.array(~i, dtype=bool))[0]
    upm = max(ii)
    upi = u[u > upm]
    lom = min(ii)
    loi = u[u < lom]

    if show:
        x = np.linspace(0, y.shape[0] - 1, y.shape[0])

        plt.figure()
        ax1 = plt.subplot2grid((1,3), (0, 0), colspan=2)
        ax1.imshow(img)
        ax1.set_xlim([0, img.shape[1] -1])
        ax2 = plt.subplot2grid((1,3), (0, 2))
        ax2.plot(y, x)
        ax2.set_ylim([0, y.shape[0] - 1])
        ax2.invert_yaxis()
        ax2.plot([ch, ch], [x[0], x[-1]])
        ax2.plot([0, ym], [ya, ya])
        ax2.plot(rmv, arm, 'kx')
        plt.show()

    sym = 0
    if len(loi) > 0:
        sym += (sum(arm <= max(loi)) > 0) or ((lom - 1) > spw)
    if len(upi) > 0:
        sym += 2 * ((sum(arm >= min(upi)) > 0) or ((len(y) - upm - 1) > spw))

    return sym


def word_symmetry2(msk, show=True):
    """
    An alternative could be to analyze the min and max x coordinates...
    """
    l = []
    for r in range(msk.shape[0]):
        row = msk[r, :]
        i = np.where(row)[0]
        if len(i) > 0:
            l.append(max(i) - min(i))
        else:
            l.append(0)

    l = np.array(l)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(l)
        fig.show()

    # TODO: not finished
