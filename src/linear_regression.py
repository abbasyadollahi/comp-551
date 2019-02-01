import numpy as np

def linear_closed_form(X, y):
    inverse_xtx = np.linalg.inv(np.matmul(X.T, X))
    w = np.matmul(np.matmul(inverse_xtx, X.T), y)
    return w

def linear_gradient_descent(X, y, w_init, decay_speed, learn_rate, min_err, max_iter, verbose=False):
    w_prev = w_init

    xtx_product = np.matmul(X.T, X)
    xty_product = np.matmul(X.T, y)

    for num_iter in range(max_iter):
        curr_learn_rate = learn_rate / (1 + decay_speed * (num_iter + 1))
        w_curr = w_prev - 2 * curr_learn_rate * (np.matmul(xtx_product, w_prev) - xty_product)
        err = np.linalg.norm(w_curr - w_prev)

        if err < min_err or num_iter >= max_iter:
            if verbose:
                print(f'Finished after {num_iter} iterations')
            break
        w_prev = w_curr

        if num_iter % (max_iter // 100) == 0 and verbose:
            print(f'Error: {err} | Learning rate: {curr_learn_rate}')

    return w_curr
