from utils import fio
from pathlib import Path
import xml.etree.ElementTree as ET
from . import node as Node
from . import edge as Edge


class Molecule:
    def __init__(self, file_number, label=None):
        self.file_number = file_number
        self.label = label
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
        #print("Parsing file: %s" % (self.file_path))
        gxlTree = ET.parse(self.file_path)
        graph = gxlTree.find('graph')
        self.id = graph.get('id')
        """
        <node id="_1">
            <attr name="symbol">
                <string>P  </string>
            </attr>
            <attr name="chem">
                <int>6</int>
            </attr>
            <attr name="charge">
                <int>0</int>
            </attr>
            <attr name="x">
                <float>3</float>
            </attr>
            <attr name="y">
                <float>0.25</float>
            </attr>
        </node>
        """
        for node in graph.findall(".//node"):
            #print("node id: %s" % (node.get('id')))
            symbol_element = node.findall("*[@name='symbol']")[0]
            chem_element = node.findall("*[@name='chem']")[0]
            charge_element = node.findall("*[@name='charge']")[0]
            x_element = node.findall("*[@name='x']")[0]
            y_element = node.findall("*[@name='y']")[0]
            node_obj = Node.Node(node.get('id'), symbol_element[0].text, int(chem_element[0].text), int(charge_element[0].text), float(x_element[0].text), float(y_element[0].text))
            self.nodes.append(node_obj)
        """
        <edge from="_1" to="_2">
            <attr name="valence">
                <int>1</int>
            </attr>
        </edge>
        """
        for edge in graph.findall(".//edge"):
            valence_element = edge.findall("*[@name='valence']")[0]
            from_node = [n for n in self.nodes if n.id == edge.get('from')][0]
            to_node = [n for n in self.nodes if n.id == edge.get('to')][0]
            edge_obj = Edge.Edge(from_node, to_node, int(valence_element[0].text))
            from_node.add_out_edge(edge_obj)
            to_node.add_in_edge(edge_obj)
            self.edges.append(edge_obj)
        
    def get_id(self):
        return self.id
    
    def get_nodes(self):
        return self.nodes
        
    def get_edges(self):
        return self.edges
        
    def get_label(self):
        return self.label