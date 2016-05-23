# Create the html display of a given search
# Use the image as background and overlay the svg
# (remove all the paths in the svg that were not found and write the copy in a new file)
import os
from svg.path import parse_path
from xml.dom.minidom import Document, parse, parseString
from utils.fio import get_absolute_path, get_config
from utils.transcription import WordCoord

class HTMLVisualization:
    def __init__(self, word):
        self.config = get_config()
        self._document = Document()
        self.html = self._document.createElement('html')
        self.head = self._document.createElement('head')
        self.body = self._document.createElement('body')
        self.html.appendChild(self.head)
        self.html.appendChild(self.body)
        title = self._document.createElement('title')
        title.appendChild(self._document.createTextNode(word))
        self.head.appendChild(title)
        empty_text = self._document.createTextNode('')
        css = self._document.createElement('link')
        css.setAttribute('rel', 'stylesheet')
        css.setAttribute('type', 'text/css')
        css.setAttribute('href', get_absolute_path(self.config.get('KWS.search', 'css')))
        css.appendChild(empty_text)
        self.head.appendChild(css)
        js = self._document.createElement('script')
        js.setAttribute('type', 'text/javascript')
        js.setAttribute('src', get_absolute_path(self.config.get('KWS.search', 'js')))
        js.appendChild(empty_text)
        self.head.appendChild(js)

    def add_image(self, img_src, svg_src, img_id=None, paths=None, words=None, all_tooltips=True):
        div = self._document.createElement('div')
        if img_id:
            div.setAttribute('id', img_id)
        div.setAttribute('style', 'background-image: url("' + img_src + '");')
        div.setAttribute('class', 'svg');
        svg_doc = parse(svg_src)
        if words is not None:
            if all_tooltips:
                svg_doc = highlight_words(svg_doc, words)
            else:
                svg_doc = filter_svg_by_id(svg_doc, words)
            div.appendChild(svg_doc)
        elif paths is not None:
            svg_doc = filter_svg(svg_doc, paths)
            div.appendChild(svg_doc)
        else:
            div.appendChild(svg_doc.documentElement)
        self.body.appendChild(div)

    def add_image_by_id(self, img_id, word_ids=None, all_tooltips=True):
        img_dir = get_absolute_path(self.config.get('KWS', 'images'))
        img_src = os.path.join(img_dir, img_id + '.jpg')
        svg_dir = get_absolute_path(self.config.get('KWS', 'locations'))
        svg_src = os.path.join(svg_dir, img_id + '.svg')
        self.add_image(img_src, svg_src, words=word_ids, img_id=img_id, all_tooltips=all_tooltips)

    def save(self, file_path='output.html'):
        file_handle = open(file_path, 'w')
        self.html.writexml(file_handle)
        file_handle.close()

def filter_svg(svg_doc, paths):
    svg_root = svg_doc.getElementsByTagName('svg')[0]
    # copy only the root node
    new_svg = svg_root.cloneNode(False)
    for element in svg_doc.getElementsByTagName('path'):
        parsed_path = parse_path(element.getAttribute('d'))
        if parsed_path in paths:
            element.setAttribute('class', 'highlight')
            new_svg.appendChild(element)

    return new_svg

def filter_svg_by_id(svg_doc, word_ids):
    svg_root = svg_doc.getElementsByTagName('svg')[0]
    # copy only the root node
    new_svg = svg_root.cloneNode(False)
    for element in svg_doc.getElementsByTagName('path'):
        if element.getAttribute('id') in word_ids:
            element.setAttribute('class', 'highlight')
            new_svg.appendChild(element)

    return new_svg

def highlight_words(svg_doc, word_ids):
    svg_root = svg_doc.getElementsByTagName('svg')[0]
    for element in svg_doc.getElementsByTagName('path'):
        if element.getAttribute('id') in word_ids:
            element.setAttribute('class', 'highlight')

    return svg_root
