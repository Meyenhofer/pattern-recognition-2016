import os
from glob import glob

import matplotlib.pyplot as plt
from skimage.io import imread

from ip.features import compute_features
from ip.preprocess import word_preprocessor
from utils.fio import get_config, get_absolute_path, parse_svg, path2polygon
from utils.image import crop
from utils.transcription import get_transcription, get_word


def write_word_features(output_file, word_id, mat, fea):
    handle = open(output_file, 'a')
    handle.write(word_id + os.linesep)
    [handle.write('%i\t' % x) for x in fea]
    handle.write(os.linesep)

    for row in mat:
        for cell in row:
            handle.write('%f\t' % cell)

        handle.write(os.linesep)

    handle.write('###' + os.linesep)
    handle.close()


def main(imgpath=None, svgpath=None, outputfile=None, retake=True, saveimgs=True):
    print('Word pre-processing')
    config = get_config()

    # create an output file
    if outputfile is None:
        txtp = get_absolute_path(config.get('KWS.features', 'file'))
    else:
        txtp = get_absolute_path(os.path.join(outputfile))

    processed = []
    if retake and os.path.exists(txtp):
        takenext = False
        for line in open(txtp, 'r'):
            line = line.strip()
            if takenext and (len(line) >= 9):
                processed.append(line.strip())
                takenext = False
            elif line == "###":
               takenext = True
    else:
        handle = open(txtp, 'w+')
        for param, value in config.items('KWS.prepro'):
            handle.write('%s: %s%s' % (param, value, os.linesep))
        for param, value in config.items('KWS.features'):
            handle.write('%s: %s%s' % (param, value, os.linesep))
        handle.write('###' + os.linesep)
        handle.close()

    # get the data
    if svgpath is None:
        svgd = get_absolute_path(config.get('KWS', 'locations'))
    else:
        svgd = get_absolute_path(svgpath)
    svgs = glob(os.path.join(svgd, '*.svg'))

    if imgpath is None:
        imgd = get_absolute_path(config.get('KWS', 'images'))
    else:
        imgd = get_absolute_path(imgpath)
    imgs = glob(os.path.join(imgd, '*.jpg'))

    # parse some parameter
    threshold = float(config.get('KWS.prepro', 'segmentation_threshold'))
    relative_height = float(config.get('KWS.prepro', 'relative_height'))
    skew_resolution = float(config.get('KWS.prepro', 'angular_resolution'))
    primary_peak_height = float(config.get('KWS.prepro', 'primary_peak_height'))
    secondary_peak_height = float(config.get('KWS.prepro', 'secondary_peak_height'))
    window_width = int(config.get('KWS.features', 'window_width'))
    step_size = int(config.get('KWS.features', 'step_size'))
    blocks = int(config.get('KWS.features', 'number_of_blocks'))
    svgs.sort()
    imgs.sort()

    for svgp, imgp in zip(svgs, imgs):
        svgid = os.path.basename(svgp).replace('.svg', '')
        imgid = os.path.basename(imgp).replace('.jpg', '')
        print('\t%s\n\t%s' % (svgp, imgp))

        if svgid != imgid:
            raise IOError('the id\'s of the image file (%s) and the svg file (%s) are not the same' % (svgid, imgid))

        trans = get_transcription(svgid)

        print('\tdoc id: %s' % svgid)
        wids, paths = parse_svg(svgp)
        img = imread(imgp)
        for wid, path in zip(wids, paths):
            print('\tword id: %s' % wid)

            if retake and (processed.count(wid) == 1):
                print('\talready processed')
                continue

            # look up the corresponding word
            if saveimgs:
                imgfile = wid
                word = get_word(wid, data=trans)
                if word is not None:
                    imgfile = word.code2string() + '_' + imgfile
            else:
                imgfile = None

            # get the word image
            poly = path2polygon(path)
            roi = crop(img, poly)

            pre, sym = word_preprocessor(roi,
                                         threshold=threshold,
                                         rel_height=relative_height,
                                         skew_res=skew_resolution,
                                         ppw=primary_peak_height,
                                         spw=secondary_peak_height,
                                         save=imgfile)

            if type(pre) is str:
                print('\tpre-processing failed\n\t\t%s' % pre)
                continue

            fea = compute_features(pre,
                                   window_width=window_width,
                                   step_size=step_size,
                                   blocks=blocks)

            write_word_features(txtp, wid, fea, [pre.shape[0], pre.shape[1], sym])
            print('...')


if __name__ == '__main__':
    main()
