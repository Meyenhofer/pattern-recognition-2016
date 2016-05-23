import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from utils.fio import get_absolute_path


def parse_log(filepath):
    pat = re.compile('(x?)\s*(\S+) -> (\S+)\s+'
                     '#train-words: (\d+)\s+'
                     '#candidates: (\d+)\s+'
                     'min-dist: (\d+)\s+'
                     'votes: (\d+)\s+'
                     'cpu-time: (\d+\.\d+)\s+'
                     'id: (\S+)')

    co = []
    y_in = []
    y_out = []
    nt = []
    nc = []
    md = []
    v = []
    cpu = []
    wid = []

    for lineN, line in enumerate(open(filepath, 'r')):
        if lineN < 6:
            continue

        mat = pat.match(line.strip())

        if mat:
            co.append(mat.group(1) != 'x')
            y_in.append(mat.group(2))
            y_out.append(mat.group(3))
            nt.append(int(mat.group(4)))
            nc.append(int(mat.group(5)))
            md.append(int(mat.group(6)))
            v.append(int(mat.group(7)))
            cpu.append(float(mat.group(8)))
            wid.append(mat.group(9))

    co = np.array(co)
    nt = np.array(nt)

    return co, y_in, y_out, nt, nc, md, v, cpu, wid


def plot_accuracy(counts, labels, ntrain, cpu):
    oaa = sum(counts) / len(counts)
    cotr = counts[ntrain > 0]
    acc = sum(cotr) / len(cotr)
    cput = sum(cpu) / 60

    print('\nOverall accuracy: %.2f' % oaa)
    print('Accuracy given at least 1 training sample: %.2f' % acc)
    print('CPU time: %0.2f min' % cput)

    d = {}
    for lbl, cor, ntr, in zip(labels, counts, ntrain):
        if lbl in d:
            d[lbl][0].append(cor)
            d[lbl][1].append(ntr)
        else:
            d[lbl] = ([cor], [ntr])

    print('')

    cte = np.array([sum(x[0]) for x in d.values()])
    num = np.array([len(x[0]) for x in d.values()])
    y = cte / num
    x = np.array([x[1][0] for x in d.values()])
    x2 = np.array([len(x) for x in d])

    fig = plt.figure()
    ax = fig.add_subplot(121)
    # ax.plot(x, y, '.')
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    sc = ax.scatter(x, y, c=z, s=100, edgecolor='')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-5, max(x) + 5])
    plt.grid()
    plt.xlabel('# training samples (for given label)')
    plt.ylabel('accuracy')
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel('label density')
    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    # cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'], update_ticks=True)

    ax2 = fig.add_subplot(122)
    xy2 = np.vstack([x2, y])
    z2 = gaussian_kde(xy2)(xy2)
    sc2 = ax2.scatter(x2, y, c=z2, s=100, edgecolor='')
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlim([-5, max(x2) + 5])
    plt.grid()
    plt.xlabel('word length')
    plt.ylabel('accuracy')
    cbar = plt.colorbar(sc2)
    cbar.ax.set_ylabel('label density')

    plt.show()


def filter_results(lbls, cnts, ntrs, wids, a_filt, n_filt):
    d = {}
    for l, i, c, n in zip(lbls, wids, cnts, ntrs):
        if l in d:
            d[l][0].append(c)
            d[l][2].append(i)
        else:
            d[l] = [[c], n, [i]]

    for l in d:
        acc = sum(d[l][0]) / len(d[l][0])
        c_estr = str(acc) + a_filt
        n_estr = str(d[l][1]) + n_filt

        if eval(c_estr) and eval(n_estr):
            print('%s, %s' % (l, d[l][2]))
            return l, d[l][2]

    return None, None


if __name__ == '__main__':
    # testing log
    p = get_absolute_path('search/16-05-22_14-53_classification.log')
    co1, y_in1, y_out1, nt1, nc1, md1, v1, cpu1, ids1 = parse_log(p)

    plot_accuracy(co1, y_in1, nt1, cpu1)
    label, wid = filter_results(y_in1, co1, nt1, ids1, '<0.1', '>40')

    # training log
    p = get_absolute_path('search/16-05-22_14-45_classification.log')
    co2, y_in2, y_out2, nt2, nc2, md2, v2, cpu2, ids2 = parse_log(p)
    plot_accuracy(co2, y_in2, nt2, cpu2)

    plot_accuracy(np.append(co1, co2), np.append(y_in1, y_in2), np.append(nt1, nt2), np.append(cpu1, cpu2))
