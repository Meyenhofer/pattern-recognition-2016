import matplotlib.pyplot as plt

from ip.preprocess import word_preprocessor
from utils.fio import get_image_roi, get_config
from utils.transcription import WordCoord


if __name__ == '__main__':
    l = ['273-09-02',
         '273-09-03',
         '273-09-04',
         '273-09-05',
         '273-09-06',
         '273-09-07',
         '273-10-01',
         '273-10-02',
         '273-11-01',
         '273-12-01']

    for c in l[2:]:
        print('processing id=%s:' % c)
        th = 0.2
        wordcoord = WordCoord(c)
        roi = get_image_roi(wordcoord)

        config = get_config()
        pre = word_preprocessor(roi,
                                float(config.get('KWS.prepro', 'segmentation_threshold')),
                                float(config.get('KWS.prepro', 'relative_height')),
                                float(config.get('KWS.prepro', 'angular_resolution')),
                                show=True)
