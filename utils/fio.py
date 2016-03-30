# file input output utilities

import os
import numpy as np
from configparser import ConfigParser as ConfigParser


def parse_mnist(filepath, numlines=np.Inf):
    lbl = []
    dat = []
    for num, line in enumerate(open(filepath)):
        parts = line.strip().split(',')
        parts = [int(x) for x in parts]
        lbl.append(parts[0])
        dat.append(parts[1:])
        if num > numlines:
            break

    return np.array(lbl), np.array(dat)


def plot_file(prefix, type='pdf'):
    config = ConfigParser()
    config.read('config.ini')

    plotdir = config.get('Plots', 'directory')
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    parts = prefix.split('.')
    type.replace('.', '')
    if len(parts) == 1:
        filename = prefix + '.' + type
    elif len(parts) == 2:
        filename = prefix
    else:
        raise RuntimeError("found multiple extensions in " + prefix)

    return os.path.join(plotdir, filename)
