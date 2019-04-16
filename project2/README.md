# Project 2

Classifying reviews from an IMBD [dataset](https://www.kaggle.com/c/comp-551-imbd-sentiment-classification/data) as positive or negative.

See `requirements.txt` for required packages.

## Linear Models

- Unzip training and test data under `data`.
- Run `src/train_models.py` to train all models and obtain testing and validation accuracy. All models get saved under `data/model/` after training.
- Use the template in `src/test_model.py`. Replace with the name of the model you want to test and enter a name for the CSV file.
- CSV file containing results will be stored under `data/result/`.

## LSTM

- Unzip training and test data under `data`.
- Run `src/lstm.py` to train model.
- Model will be saved under `data/model/lstm.h5`.
- Use Keras' `load_model` function to load the trained model and generate your predictions.
