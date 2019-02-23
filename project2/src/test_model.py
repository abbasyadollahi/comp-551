import time
from sklearn.model_selection import train_test_split

from data import load_test, predictions_to_csv, save_model, load_model
from pipeline import naive_bayes_pipeline, log_reg_pipeline, linear_svc_pipeline

if __name__ == '__main__':
	# Loading test data
	print('Loading data...')
	start = time.time()
	data_test = load_test()
	print(f'Time to load data: {time.time()-start}')

	# Replace with name of model you want load
	pipeline = load_model('sgd_bigram_tfidf.joblib')

	# Generate predictions
	pred = pipeline.predict(data_test)

	# Name of csv to save predictions in
	predictions_to_csv(pred, 'sgd_bigram_tfidf.csv')
