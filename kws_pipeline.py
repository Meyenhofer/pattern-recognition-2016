import numpy as np
from search.KNN import KNN


# Load training and validation data
knn = KNN()
knn.load_train_and_valid()

# Get one single page from the validation data
index = np.array([x.doc_id == '300' for x in knn.valid.coords], dtype=bool)
dataset = knn.valid.subset(index)

# Search
knn.set_data(dataset)
knn.create_index()
knn.search_word('the')
