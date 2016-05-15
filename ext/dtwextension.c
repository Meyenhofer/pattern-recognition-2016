#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "dtw.h"

static PyObject* dtw_extension(PyObject *dummy, PyObject *args) {
    PyObject *arg1 = NULL;
    PyObject *arr1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arr2 = NULL;

    if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
      return NULL;
    }

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) {
      return NULL;
    }
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr2 == NULL) {
      return NULL;
    }

    // Number of dimensions
    int nd1 = PyArray_NDIM((PyArrayObject*)arr1);
    int arr_type1 = PyArray_TYPE((PyArrayObject*)arr1);
    npy_intp *dims1 = PyArray_DIMS((PyArrayObject*)arr1);
    int nd2 = PyArray_NDIM((PyArrayObject*)arr2);
    int arr_type2 = PyArray_TYPE((PyArrayObject*)arr2);
    npy_intp *dims2 = PyArray_DIMS((PyArrayObject*)arr2);
    int r = 0;
    npy_double **input1 = NULL;
    r = PyArray_AsCArray((PyObject**)&arr1, (void**)&input1, dims1, nd1, PyArray_DescrFromType(arr_type1));
    if (r < 0) {
      PyErr_SetString(PyExc_RuntimeError, "Could not convert input to C array");
      return NULL;
    }
    npy_double **input2 = NULL;
    r = PyArray_AsCArray((PyObject**)&arr2, (void**)&input2, dims2, nd2, PyArray_DescrFromType(arr_type2));
    if (r < 0) {
      PyErr_SetString(PyExc_RuntimeError, "Could not convert input to C array");
      return NULL;
    }

    // DTW
    matrix_t *mat_a = malloc(sizeof(*mat_a));
    matrix_t *mat_b = malloc(sizeof(*mat_b));
    double **arr_a = alloc_2darr(dims1[0], dims1[1]);
    double **arr_b = alloc_2darr(dims2[0], dims2[1]);
    for (int i = 0; i < dims1[0]; i++) {
      memcpy(arr_a[i], input1[i], dims1[1] * sizeof(**arr_a));
    }
    for (int j = 0; j < dims2[0]; j++) {
      memcpy(arr_b[j], input2[j], dims2[1] * sizeof(**arr_b));
    }
    mat_a->arr = arr_a;
    mat_a->rows = dims1[0];
    mat_a->cols = dims1[1];
    mat_b->arr = arr_b;
    mat_b->rows = dims2[0];
    mat_b->cols = dims2[1];
    double dist = dtw_distance(mat_a, mat_b);
    free_matrix(mat_a);
    free_matrix(mat_b);
    PyObject *value = PyFloat_FromDouble(dist);
    if (value == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "Could not convert double to object");
      return NULL;
    }
    Py_DECREF(arr1);
    Py_DECREF(arr2);

    PyArray_Free(arr1, (void*)input1);
    PyArray_Free(arr2, (void*)input2);

    return value;
}

static struct PyMethodDef methods[] = {
    {"dtwdistance", dtw_extension, METH_VARARGS, "Compute DTW Distance"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "dtwextension",
  "Dynamic Time Warping C extension",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_dtwextension(void) {
    import_array();
    return PyModule_Create(&module);
}
