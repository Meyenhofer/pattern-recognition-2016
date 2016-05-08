# Create the html display of a given search
# Use the image as background and overlay the svg
# (remove all the paths in the svg that were not found and write the copy in a new file)
from svg.path import parse_path
from xml.dom.minidom import Document, parse, parseString

class HTMLVisualization:
    def __init__(self):
        self.__document = Document()
        self.html = self.__document.createElement('html')
        self.head = self.__document.createElement('head')
        self.body = self.__document.createElement('body')
        self.html.appendChild(self.head)
        self.html.appendChild(self.body)
        css = self.__document.createElement('style')
        text = self.__document.createTextNode('div { position: absolute; }')
        css.appendChild(text)
        self.head.appendChild(css)

    def add_image(self, img_src, svg_src=None, svg_str=None, img_id=None, paths=None):
        div = self.__document.createElement('div')
        if img_id:
            div.setAttribute('id', img_id)
        div.setAttribute('style', 'background-image: url("' + img_src + '");')
        if svg_src:
            svg_doc = parse(svg_src)
        elif svg_str:
            svg_doc = parseString(svg_str)
        if svg_doc:
            if paths is not None:
                svg_doc = filter_svg(svg_doc, paths)
                div.appendChild(svg_doc)
            else:
                div.appendChild(svg_doc.documentElement)
        self.body.appendChild(div)

    def save(self, file_path='output.html'):
        file_handle = open(file_path, 'w')
        self.html.writexml(file_handle)
        file_handle.close()

def filter_svg(svg_doc, paths):
    print(svg_doc, svg_doc.getElementsByTagName('svg'))
    svg_root = svg_doc.getElementsByTagName('svg')[0]
    # copy only the root node
    new_svg = svg_root.cloneNode(False)
    for element in svg_doc.getElementsByTagName('path'):
        parsed_path = parse_path(element.getAttribute('d'))
        for path in paths:
            if path == parsed_path:
                new_svg.appendChild(element)
                break

    return new_svg
