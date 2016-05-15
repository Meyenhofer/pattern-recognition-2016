#ifndef _DTW_H_
#define _DTW_H_
// Matrix structure
typedef struct matrix_t {
  void *arr;
  int rows;
  int cols;
} matrix_t;


// Matrix functions
void print_matrix(matrix_t *mat);
void free_matrix(matrix_t *mat);
double matrix_element(matrix_t *mat, int row, int col);

// 2D double array memory management
double **alloc_2darr(int rows, int cols);
void free_2darr(double **arr, int rows);

// DTW functions
matrix_t *dtw(matrix_t *x, matrix_t *y);
double dtw_distance(matrix_t *x, matrix_t *y);
double euclidean_distance(double x[], double y[], int length);
double min(double a, double b, double c);
#endif
