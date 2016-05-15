#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "dtw.h"

static PyObject* pairwise_dist(PyObject *dummy, PyObject *args) {
    PyObject *arg1 = NULL;
    PyObject *arr1 = NULL;

    if (!PyArg_ParseTuple(args, "O", &arg1)) {
      return NULL;
    }

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) {
      return NULL;
    }

    // Number of dimensions
    int nd = PyArray_NDIM((PyArrayObject*)arr1);
    int arr_type = PyArray_TYPE((PyArrayObject*)arr1);
    npy_intp *dims = PyArray_DIMS((PyArrayObject*)arr1);
    int arr_size = (dims[0] * (dims[0] - 1)) / 2;
    double rows[arr_size];
    double cols[arr_size];
    double dist[arr_size];
    int c = 0;
    int r = 0;
    npy_double ***input_array = NULL;
    r = PyArray_AsCArray((PyObject**)&arr1, (void***)&input_array, dims, nd, PyArray_DescrFromType(arr_type));
    if (r < 0) {
      PyErr_SetString(PyExc_RuntimeError, "Could not convert input to C array");
      return NULL;
    }

    // DTW
    for (int i = 0; i < dims[0] - 1; i++) {
      for (int j = i + 1; j < dims[0]; j++) {
        rows[c] = i;
        cols[c] = j;
        matrix_t *mat_a = malloc(sizeof(*mat_a));
        matrix_t *mat_b = malloc(sizeof(*mat_b));
        double **arr_a = alloc_2darr(dims[1], dims[2]);
        double **arr_b = alloc_2darr(dims[1], dims[2]);
        for (int x = 0; x < dims[1]; x++) {
          memcpy(arr_a[x], input_array[i][x], sizeof(**arr_a));
          memcpy(arr_b[x], input_array[j][x], sizeof(**arr_b));
        }
        mat_a->arr = arr_a;
        mat_a->rows = dims[1];
        mat_a->cols = dims[2];
        mat_b->arr = arr_b;
        mat_b->rows = dims[1];
        mat_b->cols = dims[2];
        dist[c] = dtw_distance(mat_a, mat_b);
        free_matrix(mat_a);
        free_matrix(mat_b);
        c += 1;
        // Print a dot for every 1000 words processed
        if (c % 1000 == 0) {
          printf(".");
        }
      }
    }

    PyObject *list = PyList_New(3);
    if (list == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "Could not create list");
      return NULL;
    }
    npy_intp obj_dims[1] = { (npy_intp)c };
    PyObject *obj1 = PyArray_SimpleNew(1, obj_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)obj1), rows, sizeof(rows));
    r = PyList_SetItem(list, 0, obj1);
    if (r < 0) {
      PyErr_SetString(PyExc_RuntimeError, "Could not append ROWS array to list");
      return NULL;
    }
    PyObject *obj2 = PyArray_SimpleNew(1, obj_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)obj2), cols, sizeof(cols));
    r = PyList_SetItem(list, 1, obj2);
    if (r < 0) {
      PyErr_SetString(PyExc_RuntimeError, "Could not append COLS array to list");
      return NULL;
    }
    PyObject *obj3 = PyArray_SimpleNew(1, obj_dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)obj3), dist, sizeof(dist));
    r = PyList_SetItem(list, 2, obj3);
    if (r < 0) {
      PyErr_SetString(PyExc_RuntimeError, "Could not append DIST array to list");
      return NULL;
    }

    Py_DECREF(arr1);

    PyArray_Free(arr1, (void*)input_array);

    return list;
}

static struct PyMethodDef methods[] = {
    {"pairwisedist", pairwise_dist, METH_VARARGS, "Pairwise distinct"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "pairwisedist",
  "Pairwise distinct",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_pairwisedist(void) {
    /* (void)Py_InitModule("pairwisedist", methods); */
    import_array();
    return PyModule_Create(&module);
}
