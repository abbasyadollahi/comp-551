#!usr/bin/python3
import numpy
from numpy.linalg import inv
def linear_closed_form(x, y):
    transpose_x = numpy.transpose(x)
    inverse_xtx_product = inv(numpy.matmul(x, transpose_x))
    closed_form_w_vector = numpy.matmul(numpy.matmul(inverse_xtx_product, transpose_x), y)
    return closed_form_w_vector
