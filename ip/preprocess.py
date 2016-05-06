import matplotlib.pyplot as plt

import os
import numpy as np
from skimage.io import imsave
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction, diamond
from skimage.transform import rescale

from ip.register import skew_correction
from utils.fio import get_image_roi, get_project_root_directory, get_config
from utils.transcription import get_transcription


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


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

    seed = np.zeros(bwcd.shape, dtype=bool)
    seed[y_lb:y_ub, x_lb:x_ub] = True
    seed &= bwcd

    rec = reconstruction(seed, bwcd)

    return np.array(rec, dtype=bool)


def standardize_roi_height(roi, standard_height=20, rel_height=0.666):
    h = standard_height * 5

    y = roi.sum(1)
    y[y < y.max() * rel_height] = 0
    nzy = y.nonzero()
    y_lb = nzy[0].min()
    y_ub = nzy[0].max()

    pw = y_ub - y_lb
    sf = standard_height / pw

    sh = int(np.ceil(float(roi.shape[0]) * pw))
    if sh <= h:
        res = rescale(roi, sf)
    else:
        if h < roi.shape[0]:
            sf = h / roi.shape[0]
            res = rescale(roi, sf)
        else:
            res = roi

    w = res.shape[1]

    c = int(h / 2)
    os_p = int(np.floor(res.shape[0] / 2))
    os_m = int(np.ceil(res.shape[0] / 2))

    uni = np.zeros((h, w))
    uni[c - os_m: c + os_p, :] = res

    return uni


def word_preprocessor(roi, threshold=0.2, rel_height=0.666, skew_res=0.33, show=False, save=None):
    # scale intensity
    ma = roi.max()
    mi = roi.min()
    roi = (roi - mi) / (ma - mi)

    # invert
    roi = roi.max() - roi

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

    # standardize the height
    # TODO this might not work as robustly as it should
    uni = standardize_roi_height(cle, standard_height=20, rel_height=rel_height)
    uni = uni / uni.max()

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
        config = get_config()
        plotdir = os.path.join(get_project_root_directory(), config.get('Plots', 'directory'))
        plotfile = os.path.join(plotdir, save + '_word-prepro.png')
        imsave(plotfile, uni)

    return uni
