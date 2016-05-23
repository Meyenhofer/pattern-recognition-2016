from search.KNN import KNN, DataSet
from utils.transcription import get_coords, get_doc_coords


def subset_test():
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
          '270-18-03']

    l3 = ['271-17-10']

    coords = get_coords('Letters')
    coords.extend(get_coords('Fort'))

    ds = DataSet.parse(id_filter=[x.id for x in coords])
    val = KNN()
    val.fit(ds)
    # val.set_tol(100, 500)
    # val.set_k(3)
    val.test(val.train.X, val.train.Y, val.train.coords, imgs=val.train.imgs)


def training_test():
    knn = KNN()
    knn.load_train_and_valid()

    # knn.set_k(3)
    # knn.set_tol(20, 200)

    # knn.test(knn.train.X, knn.train.Y, knn.train.coords, imgs=knn.train.imgs)
    knn.test(knn.train.X, knn.train.Y, knn.train.coords)


def validation_test():
    knn = KNN()
    knn.load_train_and_valid()

    # knn.set_k(3)
    # knn.set_tol(20, 300)

    # knn.test(knn.valid.X, knn.valid.Y,knn.valid.coords, imgs=knn.valid.imgs)
    knn.test(knn.valid.X, knn.valid.Y, knn.valid.coords)


if __name__ == '__main__':
    # subset_test()

    # training_test()

    validation_test()
