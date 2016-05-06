import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

from ip.preprocess import clean_crop, create_word_mask
from ip.register import skew_correction
from utils.fio import get_image_roi
from utils.transcription import WordCoord


l = ['273-06-01',
     '273-06-02',
     '273-06-03',
     '273-06-04',
     '273-06-05',
     '273-06-06',
     '273-06-07',
     '273-06-08',
     '273-07-01',
     '273-09-01']

for c in l[1:2]:
    print('processing id=%s:' % c)
    wordcoord = WordCoord(c)
    roi = get_image_roi(wordcoord)
    msk = create_word_mask(roi)
    img = np.copy(roi)
    img = img.max() - img
    img[msk < 1] = 0
    rot, ang = skew_correction(msk)
    if ang != 0:
        img = rotate(img, ang)

    cle = clean_crop(img)

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.imshow(roi)
    ax = fig.add_subplot(312)
    ax.imshow(rot)
    ax = fig.add_subplot(313)
    ax.imshow(cle)
    plt.show()
