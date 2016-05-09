import matplotlib.pyplot as plt     # For debugging

from ip.features import compute_features
from ip.preprocess import word_preprocessor
from utils.fio import get_image_roi, get_config, parse_feature_map, get_absolute_path
from utils.transcription import WordCoord


if __name__ == '__main__':

    # parse some parameter
    config = get_config()
    threshold = float(config.get('KWS.prepro', 'segmentation_threshold'))
    relative_height = float(config.get('KWS.prepro', 'relative_height'))
    skew_resolution = float(config.get('KWS.prepro', 'angular_resolution'))
    standard_height = float(config.get('KWS.prepro', 'central_height'))
    window_width = int(config.get('KWS.features', 'window_width'))
    step_size = int(config.get('KWS.features', 'step_size'))

    # some sample images
    l = ['304-13-03',
         '270-32-04',
         '277-23-01',
         '302-09-06',
         '303-29-02',
         '304-05-08']

    for c in l[:]:
        print('processing id=%s:' % c)
        th = 0.2
        wordcoord = WordCoord(c)
        roi = get_image_roi(wordcoord)

        if roi.shape[0] < standard_height:
            print('\tskipping (image height is only %s)' % roi.shape[0])
            continue

        pre = word_preprocessor(roi,
                                threshold=threshold,
                                rel_height=relative_height,
                                skew_res=skew_resolution,
                                sta_height=standard_height,
                                show=True)

        fea = compute_features(pre,
                               window_width=window_width,
                               step_size=step_size)

    # Read the feature map
    fpath = get_absolute_path('ip/feature-map.txt')
    a, b, c = parse_feature_map(fpath)
    print('')
