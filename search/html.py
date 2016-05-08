# Create the html display of a given search
# Use the image as background and overlay the svg
# (remove all the paths in the svg that were not found and write the copy in a new file)
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

    def add_image(self, img_src, svg_src=None, svg_str=None, img_id=None):
        div = self.__document.createElement('div')
        if img_id:
            div.setAttribute('id', img_id)
        div.setAttribute('style', 'background-image: url("' + img_src + '");')
        if svg_src:
            svg_doc = parse(svg_src)
        elif svg_str:
            svg_doc = parseString(svg_str)
        if svg_doc:
            div.appendChild(svg_doc.documentElement)
        self.body.appendChild(div)

    def save(self, file_path='output.html'):
        file_handle = open(file_path, 'w')
        self.html.writexml(file_handle)
        file_handle.close()
