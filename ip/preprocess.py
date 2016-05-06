# import matplotlib.pyplot as plt
# from skimage.io import imshow
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction
from skimage.transform import rotate

from utils.fio import get_image_roi
from utils.transcription import get_transcription


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def compute_central_heights():
    trc = get_transcription()
    wh = []
    for coord, word in trc:
        roi = get_image_roi(coord)
        y = roi.sum(1)
        y_max = y.max()
        y[y < y_max] = 0
        nz = y.nonzero()
        wh.append(nz[0].shape[0])

    return np.array(wh)


def clean_crop(roi, rel_height=0.333, mask=None):
    if not mask:
        mask = create_word_mask(roi, rel_height=rel_height)

    # set background to zero and normalize gray scale values [0..1]
    img = np.copy(roi)
    img[mask] = 0
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


def create_word_mask(roi, threshold=None, rel_height=0.333):
    if not threshold:
        threshold = threshold_otsu(roi)

    # binarize
    bw = roi > threshold

    # remove small objects
    lbl, _ = ndi.label(~bw)
    sizes = np.bincount(lbl.ravel())
    mask_sizes = sizes > 20
    mask_sizes[0] = 0
    bwc = mask_sizes[lbl]

    # dilate
    bwcd = ndi.binary_dilation(bwc)

    # morphological reconstruction from the center part
    # to make sure that the mask contains that every considered object
    # is in fact anchored there
    x = bwc.sum(0)
    x[x < x.max() * rel_height] = 0
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
