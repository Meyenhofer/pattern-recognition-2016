# file input output utilities

import os
import csv
import random
import numpy as np
from configparser import ConfigParser as ConfigParser
from svg.path import parse_path
from xml.dom import minidom


def parse_svg(filepath):
    doc = minidom.parse(filepath)
    paths = []
    for path in doc.getElementsByTagName('path'):
        parsed_path = parse_path(path.getAttribute('d'))
        paths.append(parsed_path)

    return np.array(paths)


def path2polygon(path):
    start = path[0].start
    polygon = [(start.imag, start.real)]
    for line in path:
        polygon.append((line.end.imag, line.end.real))

    return polygon


def parse_mnist(filepath, numlines=np.Inf):
    lbl = []
    dat = []
    numlines -= 1
    for num, line in enumerate(open(filepath)):
        parts = line.strip().split(',')
        parts = [int(x) for x in parts]
        lbl.append(parts[0])
        dat.append(parts[1:])
        if num >= numlines:
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


def get_plot_file(prefix, filetype='pdf'):
    config = get_config()
    plotdir = os.path.join(get_project_root_directory(), config.get('Plots', 'directory'))

    return get_data_location(plotdir, prefix, filetype)


def get_classifier_file(name):
    config = get_config()
    parent = os.path.join(get_project_root_directory(), config.get('Classifiers', 'directory'))

    return get_data_location(parent, name, 'plk')


def get_data_location(directory, filename, extension):
    if not os.path.exists(directory):
        os.mkdir(directory)

    parts = filename.split('.')
    extension.replace('.', '')
    if len(parts) == 1:
        filename = filename + '.' + extension
    elif len(parts) == 2:
        filename = filename
    else:
        raise RuntimeError("found multiple extensions in " + filename)

    return os.path.join(directory, filename)


def get_config():
    config = ConfigParser()
    path = 'config.ini'
    while len(config.sections()) == 0:
        config.read(path)
        path = "../" + path

    return config


def get_project_root_directory():
    config = get_config()
    pdn = config.get('PROJECT', 'directory')
    cwd = os.path.dirname(__file__)
    while os.path.basename(cwd) != pdn:
        cwd = os.path.dirname(cwd)

    return cwd


def get_absolute_path(rel_path):
    """
    Get the absolute path in the current file system from a relative path as
    they are defined in the config file
    """
    ap = get_project_root_directory()
    for i in range(rel_path.count('../')):
        ap = os.path.dirname(ap)

    return os.path.join(ap, rel_path.replace('../', ''))
