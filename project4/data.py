import os
import os.path as op
from pandas import DataFrame, read_pickle

DATA_DIR = op.abspath(op.join(__file__, op.pardir, 'data'))

def load_data(filename):
	df = read_pickle(op.join(DATA_DIR, filename))
	if df.shape[1] == 3:
		train_data = []
		train_labels = []
		dev_data = []
		dev_labels = []
		test_data = []
		test_labels = []
		for idx, row in df.iterrows():
			if row['split'] == 'train':
				train_data.append(row['sentence'])
				train_labels.append(row['label'])
			elif row['split'] == 'dev':
				dev_data.append(row['sentence'])
				dev_labels.append(row['label'])
			elif row['split'] == 'test':
				test_data.append(row['sentence'])
				test_labels.append(row['label'])
			else:
				raise Exception(f'Unknown label {row["split"]}')
		return train_data, train_labels, dev_data, dev_labels, test_data, test_labels
	else:
		return list(df.sentence), list(df.label)
