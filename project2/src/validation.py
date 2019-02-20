import os

try:
    os.chdir(os.path.join(os.getcwd(), 'project2/src'))
    print(os.getcwd())
except:
    pass

data_dir = '../data'
test_dir = os.path.join(data_dir, 'test')

def load_test(data_path):
    data = []
    for file_name in os.listdir(data_path):
        with open(os.path.join(data_path, file_name), 'rb') as f:
            review = f.read().decode('utf-8').replace('\n', '').strip().lower()
            data.append([review, int(file_name.split('.')[0])])
    return data
