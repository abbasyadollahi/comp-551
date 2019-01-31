import numpy as np

def linear_closed_form(X, y):
    transpose_x = np.transpose(X)
    inverse_xtx_product = np.linalg.pinv(np.matmul(transpose_x, X))
    closed_form_w_vector = np.matmul(np.matmul(inverse_xtx_product, transpose_x), y)
    return closed_form_w_vector

def linear_gradient_descent(X, y, w_init, decay_speed, learn_rate, min_err, max_iter):
    w_curr = w_init
    w_prev = w_init
    num_iter = 0

    x_transpose = np.transpose(X)
    xtx_product = np.matmul(x_transpose, X)
    xty_product = np.matmul(x_transpose, y)

    while True:
        curr_learn_rate = learn_rate / (1 + decay_speed*(num_iter+1))
        w_curr = w_prev - 2*curr_learn_rate*(np.matmul(xtx_product, w_prev) - xty_product)
        err = np.linalg.norm(w_curr - w_prev)**2
        if err <= min_err or num_iter > max_iter:
            break
        num_iter += 1
        if num_iter % 10 == 0:
            print(f'Error: {err}')
    return w_curr