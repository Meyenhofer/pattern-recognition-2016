from time import asctime
from timeit import default_timer as timer
from molecules import molecule,bipartite_graph,node,edge
from utils import fio
import operator


def load_data(data_path):
    print("Loading %s" % data_path)
    molecules = []
    target_values = []
    with open(data_path, "r") as data_file:
        for data in data_file:
            line_parts = data.splitlines()[0].split(" ")
            molecule_element = molecule.Molecule(line_parts[0], line_parts[1])
            molecules.append(molecule_element)
            target_values.append(line_parts[1])
    print("Found %d molecules in '%s'." % (len(molecules), data_path))
    return molecules, target_values
    
    
def load_evaluation_data(data_path):
    print("Loading %s" % data_path)
    molecules = []
    with open(data_path, "r") as data_file:
        for data in data_file:
            line = data.splitlines()[0]
            molecule_element = molecule.Molecule(line)
            molecules.append(molecule_element)
    print("Found %d molecules in '%s'." % (len(molecules), data_path))
    return molecules
    
    
def run_evaluation(test_set, train_set, k):
    print("Starting evaluation of test set (length=%d) with k=%d." % (len(test_set), k))
    start = timer()
    predictions = bipartite_graph.evaluate_test_set(test_set, train_set, k)
    accuracy = bipartite_graph.get_accuracy(predictions, test_set)
    end = timer()
    print("Duration of evaluating test set: %f" % (end - start))
    print("The accuracy for k=%d is: %f" % (k, accuracy))
    export_predictions(predictions)


def export_predictions(predictions):
    config = fio.get_config()
    file_path = config.get('molecules', 'root')
    file_path += "/molecules_result.csv"
    data = []
    for element in predictions:
        molecule = element[0]
        label = element[1]
        data_element = [molecule.get_file_number(), label]
        data.append(data_element)
    data.sort(key=operator.itemgetter(0))
    fio.export_csv_data(file_path, data)
        

def main():
    print("%s | Start running molecules_pipeline." % asctime())
    print("================================================================================")
    start = timer()
    
    #mol16 = molecule.Molecule("16")
    #print(mol16.get_id())
    #print("# nodes: %d" % (len(mol16.get_nodes())))
    #print("# edges: %d" % (len(mol16.get_edges())))
    
    #node1 = mol16.get_nodes()[0]
    #print("Node '%s' outdegree: %d" % (node1, node1.get_outdegree()))
    #print("Node '%s' indegree: %d" % (node1, node1.get_indegree()))
    
    #mol40 = molecule.Molecule("40")
    #print(mol40.get_id())
    
    #cost_matrix = bipartite_graph.build_cost_matrix(mol16, mol40)
    #print(cost_matrix)
    
    #row_ind, col_ind = bipartite_graph.get_optimal_assignment(cost_matrix)
    #print("%s; %s" % (row_ind, col_ind))
    
    #lsa_cost = bipartite_graph.get_assignment_cost(cost_matrix, row_ind, col_ind)
    #print(lsa_cost)
    
    
    config = fio.get_config()
    train_path = config.get('molecules', 'training')
    train_molecules, train_target_values = load_data(train_path)
    
    #test_path = config.get('molecules', 'testing')
    #test_molecules, test_target_values = load_data(test_path)
    #test_element = test_molecules[0]
    
    #knn_classifier = bipartite_graph.get_knn_classifier(train_molecules, train_target_values)
    #knn_result = bipartite_graph.get_k_nearest_neighbors(knn_classifier, test_element)
    
    #knn_result = bipartite_graph.get_k_nearest_neighbors(train_molecules, test_element)
    #label = bipartite_graph.determine_most_frequent_label(knn_result)
    #print("Predicted label: %s" % label)
    
    
    evaluation_path = config.get('molecules', 'evaluation')
    evaluation_molecules = load_evaluation_data(evaluation_path)
    
    run_evaluation(evaluation_molecules, train_molecules, 3)
    
    end = timer()
    print("================================================================================")
    print("Duration: %f" % (end - start))
    print("%s | End of molecules_pipeline." % asctime())


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()