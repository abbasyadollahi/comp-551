import numpy

def linear_gradientdescent(x, y, w0, beta, eta0, epsilon):
    transpose_x = numpy.transpose(x)
    i = 1
    alpha = eta0/(1 + beta*i)
    old_w = w0
    new_w = numpy.subtract(old_w, 2*alpha*numpy.subtract(numpy.matmul(numpy.matmul(transpose_x, x), old_w), numpy.matmul(transpose_x, y)))
    while numpy.absolute(numpy.subtract(old_w, new_w)) > epsilon
        alpha = eta0/(1 + beta*i)
        old_w = new_w
        new_w = numpy.subtract(old_w, 2*alpha*numpy.subtract(numpy.matmul(numpy.matmul(transpose_x, x), old_w), numpy.matmul(transpose_x, y)))
        i = i + 1
    return new_w