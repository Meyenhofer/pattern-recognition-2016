import numpy as np
import math
import operator
from scipy.optimize import linear_sum_assignment
#from sklearn.neighbors import KNeighborsClassifier


def build_cost_matrix(molecule1, molecule2, cost_node=1, cost_edge=1, cost_subst=None):
    if (cost_subst == None):
        cost_subst = 2 * cost_node
        
    nodes1 = molecule1.get_nodes()
    nodes2 = molecule2.get_nodes()
    n = len(nodes1)
    m = len(nodes2)
    
    # Initialize empty matrix [n+m][n+m]
    cost_matrix = np.zeros((n+m,n+m))
    #print(cost_matrix)
    
    # Fill upper left matrix: substitutions
    for i in range(0, n):
        for j in range(0, m):
            # Assign substitution cost if symbols are different
            if(nodes1[i].get_symbol() != nodes2[j].get_symbol()):
                cost_matrix[i][j] = cost_subst

    # Fill upper right matrix: deletions
    for i in range(0, n):
        for j in range(m, n+m):
            if(i == (j - m)):
                cost_matrix[i][j] = cost_node + (cost_edge * nodes1[i].total_edges())
            else:
                cost_matrix[i][j] = math.inf

    # Fill lower left matrix: insertions
    for i in range(n, n+m):
        for j in range(0, m):
            if((i - n) == j):
                cost_matrix[i][j] = cost_node + (cost_edge * nodes2[j].total_edges())
            else:
                cost_matrix[i][j] = math.inf
    
    # Lower right matrix already initialized with 0 values.
    #print(cost_matrix)
    return cost_matrix
    
    
def get_optimal_assignment(cost_matrix):
    """
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    """
    return linear_sum_assignment(cost_matrix)
    

def get_assignment_cost(cost_matrix, row_ind, col_ind):
    return cost_matrix[row_ind, col_ind].sum()
    
def get_optimal_assignment_cost(molecule1, molecule2):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    http://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
    """
    cost_matrix = build_cost_matrix(molecule1, molecule2)
    row_ind, col_ind = get_optimal_assignment(cost_matrix)
    lsa_cost = get_assignment_cost(cost_matrix, row_ind, col_ind)
    #print(lsa_cost)
    return lsa_cost
    
#def get_knn_classifier(train_elements, target_values, k=5):
#    knn_classifier = KNeighborsClassifier(n_neighbors=k, metric='pyfunc', func=bipartite_metric)
#    knn_classifier.fit(train_elements, target_values)
#    return knn_classifier
    
#def get_k_nearest_neighbors(fitted_knn_classifier, element):
#    k_nearest_neighbors = fitted_knn_classifier.predict(element)
#    return k_nearest_neighbors
    
def get_k_nearest_neighbors(train_set, sample, k=3):
    distances = []
    for i in range(len(train_set)):
        cost = get_optimal_assignment_cost(train_set[i], sample)
        distances.append((train_set[i], cost)) # Tuples of training set data and according distance/cost.
    
    distances.sort(key=operator.itemgetter(1)) # Sort distances.
    k_nearest_neighbors = []
    for i in range(k):
        try:
            if i < len(distances):
                distance_value = distances[i][0]
                k_nearest_neighbors.append(distance_value)
        except IndexError:
            pass # If there are less than k distances available just take those who are there.
    #print("The 1 nearest neighbor is: %s" % (k_nearest_neighbors[0]))
    return k_nearest_neighbors
    

def determine_most_frequent_label(k_nearest_neighbors):
    if len(k_nearest_neighbors) < 1:
        return ""
    label_counts = {}
    for i in range(len(k_nearest_neighbors)):
        neighbor = k_nearest_neighbors[i]
        label = neighbor.get_label()
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1 # Found new label, initialize with 1.
    # Order dictionary by value. See also here: http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value#613218
    sorted_label_counts = sorted(label_counts.items(), key=operator.itemgetter(1), reverse=True)
    most_frequent_label = sorted_label_counts[0][0]
    #print("Most frequent label is: %s" % (most_frequent_label))
    return most_frequent_label
    
    
def evaluate_test_set(test_set, train_set, k=3):
    predictions = []
    for i in range(len(test_set)):
        if i % 50 == 0:
            print("Processing element %d of %d." % (i, len(test_set)))
        k_nearest_neighbors = get_k_nearest_neighbors(train_set, test_set[i], k)
        label = determine_most_frequent_label(k_nearest_neighbors)
        predictions.append((test_set[i], label)) # Tuples of test set data and predicted label.
    return predictions
    
def get_accuracy(predictions, test_set):
    correct_classified_number = 0
    test_set_length = len(test_set)
    for i in range(len(predictions)):
        prediction = predictions[i]
        if prediction[0].get_label() == prediction[1]:
            correct_classified_number += 1
    accuracy_number = float(correct_classified_number / test_set_length) * 100.0
    return accuracy_number