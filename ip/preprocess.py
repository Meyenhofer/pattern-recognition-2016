import matplotlib.pyplot as plt

import os
import numpy as np
from skimage.io import imsave
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction, diamond
from skimage.transform import rescale

from ip.features import word_symmetry
from ip.register import skew_correction
from utils.fio import get_project_root_directory, get_config


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def clean_crop(roi, rel_height=0.5, mask=None, threshold=None):
    if mask is None:
        mask = create_word_mask(roi, threshold=threshold, rel_height=rel_height)

    # set background to zero and normalize gray scale values [0..1]
    img = np.copy(roi)
    img[~mask] = 0
    # sca = img / 255.0

    # determine bounding box
    # inv = img.max() - img
    xp = img.sum(0)
    yp = img.sum(1)
    nzxp = xp.nonzero()
    x_lb = nzxp[0][0]
    x_ub = nzxp[0][-1]
    nzyp = yp.nonzero()
    y_lb = nzyp[0][0]
    y_ub = nzyp[0][-1]

    # crop
    cro = img[y_lb:y_ub, x_lb:x_ub]

    return cro


def create_word_mask(roi, threshold=None, rel_height=0.5):
    if threshold is None:
        threshold = threshold_otsu(roi)

    # binarize
    bw = roi > threshold

    # remove small objects
    lbl, _ = ndi.label(bw)
    sizes = np.bincount(lbl.ravel())
    mask_sizes = sizes > 20
    mask_sizes[0] = 0
    bwc = mask_sizes[lbl]

    # dilate
    bwcd = ndi.binary_dilation(bwc, diamond(1))

    # morphological reconstruction from the center part
    # to make sure that the mask contains that every considered object
    # is in fact anchored there
    x = bwc.sum(0)
    x[x < x.max() * 0.1] = 0
    nzx = x.nonzero()
    x_lb = nzx[0].min()
    x_ub = nzx[0].max()

    y = bwc.sum(1)
    y[y < y.max() * rel_height] = 0
    nzy = y.nonzero()
    y_lb = nzy[0].min()
    y_ub = nzy[0].max()
    if y_lb == y_ub:
        y_lb -= 3
        y_ub += 3

    seed = np.zeros(bwcd.shape, dtype=bool)
    seed[y_lb:y_ub, x_lb:x_ub] = True
    seed &= bwcd

    rec = reconstruction(seed, bwcd)

    return np.array(rec, dtype=bool)


def standardize_roi_height(roi, standard_height=20, rel_height=0.666):
    h = standard_height * 5

    y = roi.sum(1)
    yp = y[y >= y.max() * rel_height]

    pw = yp.shape[0]
    sf = standard_height / pw

    sh = int(np.ceil(float(roi.shape[0]) * sf))
    if sh <= h:                 # if the scale factor estimation isn't off try to rescale according to the central part
        res = rescale(roi, sf)
    else:
        if h < roi.shape[0]:    # if the thing is too big, squeez it down
            sf = h / roi.shape[0]
            res = rescale(roi, sf)
        else:                   # if the scale factor estimation is off,
            res = roi           # but the image is still smaller than the standard, just center.

    # TODO: the centering should depend on the symmetry of the word (4 cases: are, gone, to, for)
    # w = res.shape[1]
    # c = int(h / 2)
    # os_p = int(np.floor(res.shape[0] / 2))
    # os_m = int(np.ceil(res.shape[0] / 2))
    # uni = np.zeros((h, w))
    # uni[c - os_m: c + os_p, :] = res

    # Pad
    zer = np.zeros((1, res.shape[1]))
    uni = np.append(zer, res, axis=0)
    uni = np.append(uni, zer, axis=0)
    uni = uni / uni.max()

    return uni


def word_preprocessor(roi, threshold=0.2, rel_height=0.666, skew_res=0.33, ppw=20, spw=13, show=False, save=None):
    # scale intensity
    ma = roi.max()
    mi = roi.min()
    roi = (roi - mi) / (ma - mi)

    # invert
    roi = roi.max() - roi
    roi.astype('float16')

    # create a mask
    msk = create_word_mask(roi, threshold=threshold, rel_height=rel_height)
    wl = msk.sum(0).nonzero()[0].shape[0]

    # correct skew
    img = np.copy(roi)
    if wl > 120:
        msk, ang = skew_correction(msk, step=skew_res)
        if ang != 0:
            img = ndi.rotate(img, ang)

    # crop
    cle = clean_crop(img, threshold=threshold, rel_height=rel_height)
    if cle.shape[0] < (ppw * 0.5):
        if save is not None:
            cle = cle / cle.max()
            save_word_image(cle, "___FAILED_" + save)

        h = cle.shape[0]
        return 'image height too small (%s)' % h, None

    # standardize the height
    uni = standardize_roi_height(cle, standard_height=ppw, rel_height=rel_height)

    # word symmetry
    sym = word_symmetry(cle, 0.5, ppw=ppw, spw=spw)
    # sym = word_symmetry2(msk, show=True)

    # plot
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(411)
        ax.imshow(roi)
        ax = fig.add_subplot(412)
        ax.imshow(img)
        ax = fig.add_subplot(413)
        ax.imshow(cle)
        ax = fig.add_subplot(414)
        ax.imshow(uni)
        plt.show()

    if save is not None:
        save_word_image(uni, str(sym) + '_' + save)      # TODO: remove sym

    return uni, sym


def save_word_image(img, wid):
    config = get_config()
    plotdir = os.path.join(get_project_root_directory(), config.get('Plots', 'directory'))
    plotfile = os.path.join(plotdir, wid + '.png')
    imsave(plotfile, img) # TODO: there is a warning about precision loss (convert from float64 to uint16)
