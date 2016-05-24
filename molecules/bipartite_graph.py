import numpy as np
import math
from scipy.optimize import linear_sum_assignment


#class Bipartite_Graph:

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