# Project 2
Classifying IMBD reviews as positive or negative.

See `requirements.txt` for required packages.

## Linear models

- Unzip training and test data under `data`.
- Run `src/train_models.py` to train all models and obtain testing and validation accuracies. All models are saved under `data/model` after training.
- Use the template in `src/test_model.py`. Replace with the name of the model you want to test and enter a name for the CSV file.
- CSV file containing results will be stored under `data/result`.

## LSTM

- Run `src/lstm.py` to train model.
- Model will be saved under `data/model/lstm.h5`.
- You can use Keras `load_model` function to load the trained model and generate your predictions.