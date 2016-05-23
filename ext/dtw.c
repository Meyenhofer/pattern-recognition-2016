#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dtw.h"

matrix_t *dtw(matrix_t *x, matrix_t *y) {
  double **distances = alloc_2darr(x->rows, y->rows);
  double **x_arr = x->arr;
  double **y_arr = y->arr;
  double cost = 0;
  distances[0][0] = cost + euclidean_distance(x_arr[0], y_arr[0], x->cols);
  for (int i = 1; i < x->rows; i++) {
    cost =  euclidean_distance(x_arr[i], y_arr[0], x->cols);
    distances[i][0] = cost + distances[i-1][0];
  }
  for (int j = 1; j < y->rows; j++) {
    cost = euclidean_distance(x_arr[0], y_arr[j], x->cols);
    distances[0][j] = cost + distances[0][j-1];
  }
  for (int i = 1; i < x->rows; i++) {
    for (int j = 1; j < y->rows; j++) {
      cost = euclidean_distance(x_arr[i], y_arr[j], x->cols);
      double min_dist = min_triple(distances[i-1][j], distances[i][j-1],
                            distances[i-1][j-1]);
      distances[i][j] = cost + min_dist;
    }
  }

  matrix_t *mat = malloc(sizeof(*mat));
  mat->arr = distances;
  mat->rows = x->rows;
  mat->cols = y->rows;

  return mat;
}

double dtw_distance(matrix_t *x, matrix_t *y) {
  matrix_t *mat = dtw(x, y);
  double result = matrix_element(mat, mat->rows - 1, mat->cols - 1);
  free_matrix(mat);

  return result;
}

double euclidean_distance(double x[], double y[], int length) {
  double result = 0.0;
  for (int i = 0; i < length; i++) {
    result += (x[i] - y[i])*(x[i] - y[i]);
  }

  return sqrt(result);
}

void free_matrix(matrix_t *mat) {
  free_2darr(mat->arr, mat->rows);
  free(mat);
}

void print_matrix(matrix_t *mat) {
  double **arr = mat->arr;
  printf("[\n");
  for (int k = 0; k < mat->rows; k++) {
    printf("  [");
    for (int l = 0; l < mat->cols - 1; l++) {
      printf("%.5f, ", arr[k][l]);
    }
    printf("%.5f]\n", arr[k][mat->cols - 1]);
  }
  printf("]\n");
}

double matrix_element(matrix_t *mat, int row, int col) {
  double **arr = mat->arr;
  if (row >= mat->rows || col >= mat->cols) {
    return -1;
  }

  return arr[row][col];
}

double **alloc_2darr(int rows, int cols) {
  double **arr = malloc(rows * sizeof(*arr));
  for (int i = 0; i < rows; i++) {
    arr[i] = malloc(cols * sizeof(**arr));
  }

  return arr;
}

void free_2darr(double **arr, int rows) {
  for (int i = 0; i < rows; i++) {
    free(arr[i]);
  }
  free(arr);
}

double min_triple(double a, double b, double c) {
  double min_ab = (a < b) ? a : b;
  return (min_ab < c) ? min_ab : c;
}
