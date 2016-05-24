class Node:
    def __init__(self, id, symbol, chem, charge, x, y):
        self.id = id
        self.symbol = symbol
        self.chem = chem
        self.charge = charge
        self.x = x
        self.y = y
        self.out_edges = []
        self.in_edges = []
        
    def add_out_edge(self, out_edge):
        self.out_edges.append(out_edge)
    
    def add_in_edge(self, in_edge):
        self.in_edges.append(in_edge)
    
    def get_outdegree(self):
        return len(self.out_edges)
        
    def get_indegree(self):
        return len(self.in_edges)
    
    def total_edges(self):
        return self.get_outdegree() + self.get_indegree()
    
    def get_id(self):
        return self.id
    
    def get_symbol(self):
        return self.symbol
    
    def __str__(self):
        return self.id.__str__()