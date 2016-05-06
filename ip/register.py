import numpy as np
from skimage.transform import rotate


def vertical_projection_width(img):
    y = img.sum(1)
    dy = y[0:-2] - y[1:-1]
    nz = y.nonzero()
    return nz[0].shape[0], np.absolute(dy).sum()


def iteratively_rotate(img, angle_step=0.1, max_iter=3):
    w0, d0 = vertical_projection_width(img)
    w_pre = 100000000
    d_pre = 0
    angle = angle_step
    rot_pre = img
    iter = 0
    while iter < max_iter:
        rot = rotate(img, angle, resize=False)
        w_cur, d_cur = vertical_projection_width(rot)
        if w_pre < w_cur or d_pre > d_cur:
            break

        print('\tangle=%0.1f, width=%.2f, grad=%.2f, iter=%d' % (angle, w_cur, d_cur, iter))

        if w_cur >= w0:
            iter += 1
        else:
            rot_pre = rot

        w_pre = w_cur
        d_pre = d_cur
        angle += angle_step

    return rot_pre, angle, w_pre, d_pre


def skew_correction(img, step=0.5, max_iter=5):
    w0, d0 = vertical_projection_width(img)
    print('\twidth=%s, grad=%s' % (w0, d0))
    print('\tcounter-clock-wise rotation')
    rotf, af, wf, df = iteratively_rotate(img, angle_step=step, max_iter=max_iter)

    print('\tclock-wise rotation')
    rotb, ar, wr, dr = iteratively_rotate(img, angle_step=-step, max_iter=max_iter)

    if (wf < w0 or df < d0) and (wf <= wr and df < dr):
        return rotf, af
    if wr < w0 or dr < d0:
        return rotb, -ar

    return img, 0