import os
from glob import glob

from datetime import datetime
from skimage.io import imread

from ip.features import compute_features
from ip.preprocess import word_preprocessor
from utils.fio import get_config, get_absolute_path, parse_svg, path2polygon
from utils.image import crop


def write_word_features(output_file, word_id, mat):
    handle = open(output_file, 'a')
    handle.write(word_id + os.linesep)
    for row in mat:
        for cell in row:
            handle.write('%d\t' % cell)

        handle.write(os.linesep)

    handle.write('###')
    handle.close()


def main():
    print('Word pre-processing')
    config = get_config()

    # create an output file
    txtp = get_absolute_path(config.get('KWS.features', 'file'))
    fp = os.path.join(os.path.dirname(txtp),
                      datetime.now().strftime('%y-%m-%d_%H-%M') + os.path.basename(txtp))
    handle = open(fp, 'w+')
    handle.close()

    # get the data
    svgd = get_absolute_path(config.get('KWS', 'locations'))
    svgs = glob(os.path.join(svgd, '*.svg'))
    imgd = get_absolute_path(config.get('KWS', 'images'))
    imgs = glob(os.path.join(imgd, '*.jpg'))

    for svgp, imgp in zip(svgs, imgs):
        svgid = os.path.basename(svgp).replace('.svg', '')
        imgid = os.path.basename(imgp).replace('.jpg', '')
        print('\t%s\n\t%s' % (svgp, imgp))

        if svgid != imgid:
            raise IOError('the id\'s of the image file (%s) and the svg file (%s) are not the same' % (svgid, imgid))

        print('\tdoc id: %s' % svgid)
        wids, paths = parse_svg(svgp)
        img = imread(imgp)
        for wid, path in zip(wids, paths):
            print('\tword id: %s' % wid)

            # get the word image
            poly = path2polygon(path)
            roi = crop(img, poly)

            pre = word_preprocessor(roi,
                                    float(config.get('KWS.prepro', 'segmentation_threshold')),
                                    float(config.get('KWS.prepro', 'relative_height')),
                                    float(config.get('KWS.prepro', 'angular_resolution')),
                                    save=wid)

            fea = compute_features(pre,
                                   int(config.get('KWS.features', 'window_width')),
                                   int(config.get('KWS.features', 'step_size')))

            write_word_features(fp, wid, fea)
            print('...')


if __name__ == '__main__':
    main()
