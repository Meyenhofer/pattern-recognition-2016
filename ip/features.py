from scipy.stats import moment


def compute_features(roi, window_width=1, step_size=3):
    w = roi.shape[1]
    msk = roi > 0

    f = []

    x1 = 0
    x2 = x1 + window_width
    while True:

        bin = msk[:, x1:x2]
        scv = roi[:, x1:x2]
        # gradient
        gra = bin[0:-2] - bin[1:-1]
        # black-white transitions
        bwt = gra == -1
        # white-black transitions
        wbt = gra == 1
        # digits not on the contour
        dk = gra == 0

        # foreground fraction
        fgf = bin.sum() / len(bin)

        fv = [fgf,
              bwt.sum(),
              wbt.sum(),
              dk.sum(),
              moment(scv, moment=1)[0],
              moment(scv, moment=2)[0],
              moment(scv, moment=3)[0],
              moment(scv, moment=4)[0]]

        f.append(fv)

        x1 = x2 + step_size
        x2 = x1 + window_width
        if x2 > w:
            break

    return f
