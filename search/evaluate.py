import os
import numpy as np

from ip import doc_processor
from search.KNN import DataSet
from utils.fio import get_config, get_absolute_path
from utils.transcription import Word, WordCoord
from dtwextension import dtwdistance

config = get_config()

doc_processor.main(imgpath=config.get('Evaluation.KWS', 'images'),
                   svgpath=config.get('Evaluation.KWS', 'svg'),
                   outputfile=config.get('Evaluation.KWS', 'feature-map'))


# parse the keywords
kwp = get_absolute_path(config.get('Evaluation.KWS', 'keywords'))
words = []
coords = []
for line in open(kwp, 'r'):
    parts = line.strip().split(',')
    words.append(Word(parts[0].strip()))
    coords.append(WordCoord(parts[1].strip()))

# parse the feature maps
train = DataSet.parse(get_absolute_path(config.get('KWS.features', 'file')))
evalu = DataSet.parse(get_absolute_path(config.get('Evaluation.KWS', 'feature-map')))

outputfile = os.path.join(get_absolute_path(config.get('Evaluation', 'output')), 'kws-dists.csv')
handle = open(outputfile, 'w+')
for coord, word in zip(coords, words):
    index = np.array([x.id == coord.id for x in train.coords], dtype=bool)
    key = train.subset(index)
    print('processing keyword %s (N=%i)' % (word, key.N))
    handle.write(word.code2string())

    for x, c in zip(evalu.X, evalu.coords):
        d = dtwdistance(key.X[0], x)
        handle.write(', %s, %3f' % (c.id, d))

    handle.write('\n')

handle.close()
