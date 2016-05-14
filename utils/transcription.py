from collections import OrderedDict

from utils.fio import get_config, get_absolute_path


class WordCoord:
    def __init__(self, coordinate_string):
        self.id = coordinate_string
        parts = coordinate_string.split('-')
        self.doc_id = parts[0].strip()
        self.line_id = parts[1].strip()
        self.word_id = parts[2].strip()

    def get_word(self):
        return self.word_id

    def get_line(self):
        return self.line_id

    def get_doc(self):
        return self.doc_id

    def __str__(self):
        return self.id


class Word:
    def __init__(self, word):
        self.code = OrderedDict([('-', ''),   # replace all the separator dashes
                                 ('s_pt', '.'),
                                 ('s_cm', ','),
                                 ('s_sq', ';'),
                                 ('s_qt', '\''),
                                 ('s_mi', '-'),
                                 ('s_qo', ':'),
                                 ('s_', '')])  # The last one is for number prefixes

        self.word = word

    def get_word_code(self):
        return self.word

    def code2string(self):
        s = self.word
        for key in self.code:
            s = s.replace(key, self.code[key])

        return s

    def __str__(self):
        return self.code2string()


def get_transcription(did=None):
    config = get_config()
    trans = []
    path = get_absolute_path(config.get('KWS', 'transcription'))
    for line in open(path):
        parts = line.strip().split(' ')
        coord = WordCoord(parts[0])

        if not did or coord.get_doc() == did:
            trans.append((coord, Word(parts[1])))

    trans = sorted(trans, key=lambda x: x[0].__str__())

    return trans


def get_word(coord, data=None):
    if type(coord) == WordCoord:
        coord = coord.id

    if data is None:
        data = get_transcription(coord.doc_id)

    for co, wo in data:
        if co.id == coord:
            return wo


def get_coords(word, data=None):
    if type(word) == Word:
        word = word.code2string()

    if data is None:
        data = get_transcription()

    coords = []
    for co, wo in data:
        if wo.code2string() == word:
            coords.append(co)

    return coords
