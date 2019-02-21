import os
import os.path as op

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))
TEST_DIR = op.join(DATA_DIR, 'test')
TRAIN_DIR = op.join(DATA_DIR, 'train')

def load_test():
    data = []
    for file_name in os.listdir(TEST_DIR):
        with open(os.path.join(TEST_DIR, file_name), 'rb') as f:
            review = f.read().decode('utf-8').replace('\n', '').strip().lower()
            data.append([review, int(file_name.split('.')[0])])
    return data


def load_train():
    data = []
    for category in os.listdir(TRAIN_DIR):
        category_bin = 1 if category == 'pos' else 0
        for file_name in os.listdir(os.path.join(TRAIN_DIR, category)):
            with open(os.path.join(TRAIN_DIR, category, file_name), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').strip().lower()
                data.append([review, category_bin])
    return data
