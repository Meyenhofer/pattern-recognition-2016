import numpy as np
from scipy.stats import moment


def compute_features(roi, window_width=1, step_size=3):
    w = roi.shape[1]
    msk = roi > 0
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

        # gradient
        gra = bw[0:-1] - bw[1:]
        # number of black-white and white-black transitions
        bwt, wbt = transitions(gra)
        # digits not on the contour
        dk = gra == 0
        # foreground fraction
        fgf = bw.sum() / len(bw)

        fv = [fgf,
              bwt,
              wbt,
              dk.sum(),
              np.mean(gsm),
              moment(gsm, moment=2)[0],
              moment(gsm, moment=3)[0],
              moment(gsm, moment=4)[0]]

        f.append(fv)

        x1 = x2 + step_size
        x2 = x1 + window_width
        if x2 > w:
            break

    return f


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


def word_symmetry(img, rel_height=0.66, sta_height=20):
    """
    returns [1...4]
    1: lowercase like a, e, i, o, u
    2: letters like b, t, h,
    3: letters like g, p, q
    4: letters like f
    """
    y = img.sum(1)
    i = y > y.max() * rel_height
    p = y[i]
    px = y.argmax()

    u = np.where(~i)
    up = u < px
    up = up.shape[0]
    lo = u > px
    lo = lo.shape[0]

    if y.shape[0] <= sta_height * 1.5:
        return 1
    elif (up > 10) and (lo > 10):
        return 4
    elif up > 10:
        return 2
    else:
        return 3
