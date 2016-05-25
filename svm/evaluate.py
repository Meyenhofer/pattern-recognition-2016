import os
import numpy as np
from sklearn.externals import joblib
from skimage.io import imsave

from utils.fio import get_config, get_absolute_path, get_classifier_file


def parse():
    config = get_config()
    datapath = get_absolute_path(config.get('Evaluation.SVM', 'mnist'))
    dat = []
    for line in open(datapath, 'r'):
        parts = line.strip().split(',')
        parts = [int(x) for x in parts]
        dat.append(parts)

    return np.array(dat)


def write_images():
    conf = get_config()
    odir = os.path.join(get_absolute_path(conf.get('Evaluation', 'output')), 'digits')
    if not os.path.exists(odir):
        os.mkdir(odir)

    vecs = parse()
    for i, vec in enumerate(vecs):
        d = np.math.sqrt(vec.shape[0])
        r = vec.reshape([d, d])
        n = i + 1
        fn = os.path.join(odir, '%05d.jpg' % n)
        imsave(fn, r)


def main():
    config = get_config()
    kernels = str(config.get('SVM', 'kernels')).split(',')

    outdir = get_absolute_path(config.get('Evaluation', 'output'))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dat = parse()

    for param in kernels:
        name = 'svn_' + param + '_ts-26999'
        path = get_classifier_file(name)
        print('loading classifier %s' % name)
        clf = joblib.load(path)

        print('\tpredict...')
        lbls = clf.predict(dat)

        print('\twrite output...')
        filename = os.path.join(outdir, 'svm_' + param + '.csv')
        handle = open(filename, 'w+')
        for lbl in lbls:
            handle.write('%i\n' % lbl)

        handle.close()


if __name__ == '__main__':
    # write_images()
    main()
