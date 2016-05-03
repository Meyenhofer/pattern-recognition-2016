import os
from utils import fio
from sklearn import svm, cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from pandas import DataFrame
import numpy as np
import matplotlib

from utils.fio import get_project_root_directory, get_absolute_path

matplotlib.use('Agg')   # Avoid gui.
import matplotlib.pyplot as plt


def test_kernel(x_tr, y_tr, x_te, y_te, param):
    # handle polynomial degree parameter
    degree = 3
    if param.find('poly') > -1:
        parts = param.split(' ')
        kernel = parts[0].strip()
        degree = int(parts[1].strip())
    else:
        kernel = param.strip()

    # train
    name = 'svn_' + param + '_ts-' + str(x_tr.shape[0])
    path = fio.get_classifier_file(name)
    if os.path.exists(path):
        print('      loading...')
        clf = joblib.load(path)
    else:
        print('      computing...')
        clf = svm.SVC(kernel=kernel, degree=degree)
        clf.fit(x_tr, y_tr)
        joblib.dump(clf, path)

    # score
    y_pr = clf.predict(x_te)
    score_tr = clf.score(x_tr, y_tr)
    cross_tr = np.mean(cross_validation.cross_val_score(clf, x_tr, y_tr, cv=3))
    score_te = clf.score(x_te, y_te)
    cross_te = np.mean(cross_validation.cross_val_score(clf, x_te, y_te, cv=3))
    cm = confusion_matrix(y_te, y_pr)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    cls = set(y_tr)
    ticks = np.arange(len(cls))
    labels = [str(x) for x in cls]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion matrix (kernel: %s)' % param)
    plt.xticks(ticks, labels)
    plt.ylabel('true label')
    plt.yticks(ticks, labels)
    plt.xlabel('predicted label')
    plt.tight_layout()

    path = fio.get_plot_file('SVM_confusion-matrix_' + param.replace(' ', '_'))
    fig.savefig(path)
    plt.close()

    return [score_tr, cross_tr, score_te, cross_te]


def run():
    # Get parameter from the config file
    config = fio.get_config()
    kernels = str(config.get('SVM', 'kernels')).split(',')
    train_n = int(config.get('MNIST.sample.size', 'training'))
    test_n = int(config.get('MNIST.sample.size', 'testing'))

    # Read the data

    y_train, x_train = fio.parse_mnist(get_absolute_path(config.get('MNIST', 'trainingset')), numlines=train_n)
    train_n = y_train.shape[0]
    y_test, x_test = fio.parse_mnist(get_absolute_path(config.get('MNIST', 'testset')), numlines=test_n)
    test_n = y_test.shape[0]

    print('SVN on MNIST dataset')
    print('   training set:')
    print('      # samples %s' % train_n)
    print('      # classes: %s' % len(set(y_train)))
    print('   test set:')
    print('      # testing samples %s' % test_n)
    print('      # classes %s' % len(set(y_test)))

    # Test different kernels
    scores = []
    for kernel in kernels:
        print('   kernel: %s' % kernel)
        score = test_kernel(x_train, y_train, x_test, y_test, kernel)
        print('      training score: %s' % score[0])
        print('      training cross validation %s' % score[1])
        print('      test score: %s' % score[2])
        print('      test cross validation %s' % score[3])
        scores.append(score)

    # plot the results
    df = DataFrame(np.array(scores).transpose(), columns=kernels)
    ax = df.plot.bar()
    ax.set_xticklabels(['train score', 'train: cross-val.', 'test: score', 'test: cross-val.'], rotation=0)
    ax.set_title('SVM classification (N-training = %s, N-test = %s)' % (train_n, test_n))
    ax.grid()
    ax.grid(which='minor')
    # ax.legend(loc=1)
    fig = ax.get_figure()
    path = fio.get_plot_file('SVM-scores')
    fig.savefig(path)


if __name__ == '__main__':
    run()
