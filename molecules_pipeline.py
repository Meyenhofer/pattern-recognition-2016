from time import asctime
from timeit import default_timer as Timer
from molecules import molecule


def main():
    print("%s | Start running molecules_pipeline." % asctime())
    print("================================================================================")
    start = Timer()
    
    mol16 = molecule.Molecule("16")
    print(mol16.get_id())
    print("# nodes: %d" % (len(mol16.get_nodes())))
    print("# edges: %d" % (len(mol16.get_edges())))
    
    node1 = mol16.get_nodes()[0]
    print("Node '%s' outdegree: %d" % (node1, node1.get_outdegree()))
    print("Node '%s' indegree: %d" % (node1, node1.get_indegree()))
    
    end = Timer()
    print("================================================================================")
    print("Duration: %f" % (end - start))
    print("%s | End of molecules_pipeline." % asctime())


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()