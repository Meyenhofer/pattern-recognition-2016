from search.KNN import KNN


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
      '273-34-07',
      '273-34-08',
      '273-34-09',
      '273-34-10',
      '273-35-01',
      '273-35-02',
      '273-35-03',
      '273-35-04',
      '273-35-05',
      '273-35-09',
      '274-07-06',
      '274-13-03',
      '270-27-04',
      '270-01-05'
      '273-05-07',
      '273-26-03',
      '277-24-06',
      '274-15-02',
      '270-18-03',
]

val = KNN()
val.parse(id_filter=['270-01-03'])

knn = KNN()
# knn.parse(id_filter=l2)
# knn.parse(items=20)
knn.parse_all()
knn.set_k(15)
knn.set_tol(3, 11)

# knn.test(knn.train.X, knn.train.Y, imgs=knn.train.imgs, coords=knn.train.coords)
knn.test(val.train.X, val.train.Y, imgs=val.train.imgs, coords=val.train.coords)
# knn.training_score()
# knn.test(knn.valid.X, knn.valid.Y, imgs=knn.valid.imgs, coords=knn.valid.coords)
