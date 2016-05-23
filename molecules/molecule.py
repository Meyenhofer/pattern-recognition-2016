from utils import fio
from pathlib import Path
import xml.etree.ElementTree as ET


class Molecule:
    def __init__(self, file_number):
        self.file_number = file_number
        self.nodes = []
        self.edges = []
        self.load_input_file()
   
    def load_input_file(self):
        # Get the configurations
        config = fio.get_config()
        images_path = config.get('molecules', 'images')
        self.file_path = Path(images_path + "/" + self.file_number + ".gxl").__str__()
        self.parse_gxl_file()
        
    def parse_gxl_file(self):
        """
        See also here: http://stackoverflow.com/questions/34954608/parsing-gxl-in-python/34960625
        """
        print("Parsing file: %s" % (self.file_path))
        gxlTree = ET.parse(self.file_path)
        graph = gxlTree.find('graph')
        self.id = graph.get('id')
        for node in graph.findall(".//node"):
            #print("node id: %s" % (node.get('id')))
            symbol_element = node.findall("*[@name='symbol']")[0]
            node_obj = Node(node.get('id'))
            self.nodes.append(node_obj)
        
    def get_id(self):
        return self.id
    
    def get_nodes(self):
        return self.nodes
        
        
class Node:
    def __init__(self, id, symbol="", chem="", charge="", x="", y=""):
        self.id = id
        self.symbol = symbol
        self.chem = chem
        self.charge = charge
        self.x = x
        self.y = y
        
    def __str__(self):
        return self.id.__str__()
        

class Edge:
    def __init__(self, from_id, to_id, valence):
        self.from_id = from_id
        self.to_id = to_id
        self.valence = valence