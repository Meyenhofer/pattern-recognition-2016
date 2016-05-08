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
        # number of black-white and white-black transitions
        bwt, wbt = transitions(gra)
        # digits not on the contour
        dk = gra == 0

        # foreground fraction
        fgf = bin.sum() / len(bin)

        fv = [fgf,
              bwt,
              wbt,
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
