# file input output utilities

import os
import csv
import random
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


def import_csv_data(filepath):
    csv_data = []
    with open(filepath, 'r') as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            rowArray = np.asarray(row, dtype=np.int16)
            csv_data.append(rowArray)
    return np.array(csv_data)


def split_labels_data(np_data, label_index):
    labels = []
    data = []
    for observation in np_data:
        labels.append(observation[label_index])
        data.append(np.delete(observation, label_index))
    return np.array(labels), np.array(data)


def export_csv_data(filepath, data):
    with open(filepath, "w", newline="") as file_out:
        writer = csv.writer(file_out, delimiter=',')
        if type(data) == np.ndarray:
            writer.writerows(data.tolist())
        else:
            writer.writerows(data)
    return


def get_random_data_sample(data, sample_size):
    if type(data) == np.ndarray:
        sample_data = random.sample(data.tolist(), sample_size)
    else:
        sample_data = random.sample(data, sample_size)
    return np.array(sample_data)


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
