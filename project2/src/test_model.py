import time
from sklearn.model_selection import train_test_split

from data import load_test, predictions_to_csv, save_model, load_model
from pipeline import naive_bayes_pipeline, log_reg_pipeline, linear_svc_pipeline

if __name__ == '__main__':
	# Loading all files as training data
	print('Loading data...')
	start = time.time()
	data_test = load_test()
	print(f'Time to load data: {time.time()-start}')

	print(len(data_test))

	pipeline = load_model('linsvc_bigram_tfidf.joblib')
	pred = pipeline.predict(data_test)
	print(pred.shape)
	print(pred[0])
	predictions_to_csv(pred, 'linsvc_bigram_tfidf.csv')
