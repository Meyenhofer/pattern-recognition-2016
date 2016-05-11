from search.KNN import KNN


knn = KNN()
# knn.parse_train_and_valid()
# print(knn._train.N)
# print(knn._valid.N)

l = ['304-13-03',
     '270-32-04',
     '277-23-01',
     '302-09-06',
     '303-29-02',
     '304-05-08']

l2 = ['273-34-03',
      '273-34-04',
      '273-34-05',
      '273-34-06',
      # '273-34-07',
      # '273-34-08',
      # '273-34-09',
      # '273-34-10',
      # '273-35-01',
      # '273-35-02',
      '273-35-03',
      '273-35-04',
      '273-35-05',
      '273-35-09',
      '274-07-06',
      '274-13-03']


knn.parse(id_filter=l2)
print(knn._train.N)
# print(len(l2))
# print(knn._valid.N)

# knn.parse(items=20)
print('words: ', end='')
[print(word, end='\t') for word in knn._train.words]
print('')

knn.set_k(5)
ts = knn.training_score()
print(ts)
