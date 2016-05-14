import matplotlib.pyplot as plt     # For debugging

from ip.features import compute_features, word_symmetry
from ip.preprocess import word_preprocessor
from utils.fio import get_image_roi, get_config, parse_feature_map, get_absolute_path
from utils.transcription import WordCoord


if __name__ == '__main__':

    # parse some parameter
    config = get_config()
    threshold = float(config.get('KWS.prepro', 'segmentation_threshold'))
    relative_height = float(config.get('KWS.prepro', 'relative_height'))
    skew_resolution = float(config.get('KWS.prepro', 'angular_resolution'))
    primary_peak_height = float(config.get('KWS.prepro', 'primary_peak_height'))
    secondary_peak_height = float(config.get('KWS.prepro', 'secondary_peak_height'))
    window_width = int(config.get('KWS.features', 'window_width'))
    step_size = int(config.get('KWS.features', 'step_size'))

    # some sample images
    l = ['304-13-03',
         '270-32-04',
         '277-23-01',
         '302-09-06',
         '303-29-02',
         '304-05-08']

    l2 = ['270-16-01',
          '277-05-03',
          '271-04-02',
          '270-04-05',
          # '278-01-07',
          # '272-17-06',
          # '304-19-02',
          # '274-12-07',
          # '300-27-01',
          # '300-25-05',
          # '301-27-09',
          '303-33-06']

    for c in l2[:]:
        print('processing id=%s:' % c)
        wordcoord = WordCoord(c)
        roi = get_image_roi(wordcoord)

        if roi.shape[0] < (primary_peak_height * 0.5):
            h = roi.shape[0]
            print('\tskipping (image height is only %s)' % h)
            continue

        pre, sym = word_preprocessor(roi,
                                threshold=threshold,
                                rel_height=relative_height,
                                skew_res=skew_resolution,
                                ppw=primary_peak_height,
                                spw=secondary_peak_height,
                                show=False)

        sym = word_symmetry(pre, ppw=primary_peak_height,
                                 spw=secondary_peak_height,
                                 show=True)

        fea = compute_features(pre,
                               window_width=window_width,
                               step_size=step_size)

    # Read the feature map
    fpath = get_absolute_path('ip/feature-map.txt')
    a, b, c = parse_feature_map(fpath)
    print('')
