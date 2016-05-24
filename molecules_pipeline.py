from time import asctime
from timeit import default_timer as Timer
from molecules import molecule,bipartite_graph


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
    
    mol40 = molecule.Molecule("40")
    print(mol40.get_id())
    
    cost_matrix = bipartite_graph.build_cost_matrix(mol16, mol40)
    print(cost_matrix)
    
    row_ind, col_ind = bipartite_graph.get_optimal_assignment(cost_matrix)
    print("%s; %s" % (row_ind, col_ind))
    
    lsa_cost = bipartite_graph.get_assignment_cost(cost_matrix, row_ind, col_ind)
    print(lsa_cost)
    
    end = Timer()
    print("================================================================================")
    print("Duration: %f" % (end - start))
    print("%s | End of molecules_pipeline." % asctime())


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()