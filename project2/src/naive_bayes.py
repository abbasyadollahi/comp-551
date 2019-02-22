import numpy as np

class NaiveBayes:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, x, y):
        self.X = self.vectorizer.fit_transform(x)
        self.y = np.array(y)

        self.prob_neg = (self.y == 0).sum() / self.y.size
        self.prob_pos = (self.y == 1).sum() / self.y.size

        self.mean_vector_neg = self.X[y == 0].mean(0)
        self.mean_vector_pos = self.X[y == 1].mean(0)

        self.var_vector_neg = self.X[y == 0].var(0)
        self.var_vector_pos = self.X[y == 1].var(0)

        self.neg_occurences = {0: (self.X[y == 0] == 0).sum(0), 1: (self.X[y == 0] == 1).sum(0)}
        self.pos_occurences = {0: (self.X[y == 1] == 0).sum(0), 1: (self.X[y == 1] == 1).sum(0)}

    def compute_feature_pdf(self, x, mean, var):
        return np.exp(-(x-mean)**2 / (2*var)) / np.sqrt(2 * np.pi * var) if var else 1

    def compute_negative_pdf_probability(self, dp):
        feature_pdfs = [self.compute_feature_pdf(ft, m, v) for ft, m, v in zip(dp, self.mean_vector_neg, self.var_vector_neg)]
        return self.prob_neg * np.prod(feature_pdfs)

    def compute_positive_pdf_probability(self, dp):
        feature_pdfs = [self.compute_feature_pdf(ft, m, v) for ft, m, v in zip(dp, self.mean_vector_pos, self.var_vector_pos)]
        return self.prob_pos * np.prod(feature_pdfs)

    def compute_negative_laplace_probability(self, dp):
        return self.prob_neg * np.prod([(self.neg_occurences[ft][i]+1) / (self.test_size*self.prob_neg+2) for i, ft in enumerate(dp)])

    def compute_positive_laplace_probability(self, dp):
        return self.prob_pos * np.prod([(self.pos_occurences[ft][i]+1) / (self.test_size*self.prob_pos+2) for i, ft in enumerate(dp)])

    def predict(self, dp, pdf=True):
        neg = self.compute_negative_pdf_probability(dp) if pdf else self.compute_negative_laplace_probability(dp)
        pos = self.compute_positive_pdf_probability(dp) if pdf else self.compute_positive_laplace_probability(dp)
        return 0 if neg > pos else 1

    def score(self, X, y):
        self.test_size = y.size
        predictions = [self.predict(dp, False) for dp in X]
        return sum([1 if p == r else 0 for p, r in zip(predictions, y)]) / y.size

# import math
# import numpy as np
# from scipy import sparse

# class NaiveBayes():
#     def __init__(self):
#         self.class_probabilities = {0: 0, 1: 0}
#         self.feature_probabilities = {0: [], 1: []}

#     def fit(self, X, y):
#         if isinstance(X, list):
#             X = np.array(X)
#         n, m = X.shape
#         class_counts = {0: 0, 1: 0}
#         feature_counts = {0: np.zeros(m), 1: np.zeros(m)}

#         for y_i in y:
#             class_counts[y_i] += 1

#         sparse_matrix = sparse.csr_matrix(X).nonzero()
#         (row, col) = sparse_matrix
#         for i in range(len(row)):
#             c = y[row[i]]
#             feature_counts[c][col[i]] += 1

#         self.class_probabilities = {0: math.log(class_counts[0]/float(n)), 1: math.log(class_counts[1]/float(n))}
#         self.feature_probabilities = {
#             0: [math.log((feature_count + 1)/float(class_counts[0] + 2)) for feature_count in feature_counts[0]],
#             1: [math.log((feature_count + 1)/float(class_counts[1] + 2)) for feature_count in feature_counts[1]]
#         }

#         return self

#     def predict(self, X):
#         if isinstance(X, sparse.csr.csr_matrix):
#             X = X.toarray()

#         predictions = []
#         for i, x_i in enumerate(X):
#             features = [j for j in range(len(x_i)) if x_i[j] == 1]
#             prob_1 = self.class_probabilities[1] + sum([self.feature_probabilities[1][i] for i in features])
#             prob_0 = self.class_probabilities[0] + sum([self.feature_probabilities[0][i] for i in features])

#             if prob_1 >= prob_0:
#                 predictions.append(1)
#             if prob_1 < prob_0:
#                 predictions.append(0)

#         return predictions

# from sklearn.base import BaseEstimator
# import numpy as np

# class NaiveBayes(BaseEstimator):
#     """A custom Bernoulli Naive Bayes implementation for COMP551 Mini-Project 2"""

#     def __init__(self, laplaceSmoothing=True):
#         """
#         Initializing the custom, from scratch Bernoulli Naive Bayes
#         """
#         self.laplaceSmoothing = laplaceSmoothing

#     def fit(self, X, y):
#         """
#         Fits Bernoulli Naive Bayes model to training data, X, with target values, y
#         """

#         ## number of 0/1 examples in training
#         Ny1 = np.sum(y)
#         Ny0 = y.shape[0] - Ny1

#         ## percentage of 0/1 examples in training
#         self.theta0 = Ny0/y.shape[0]
#         self.theta1 = Ny1/y.shape[0]

#         ## counts for each feature
#         Nj1 = X.T.dot(y.reshape([-1,1]))
#         Nj0 = X.T.dot(1-y.reshape([-1,1]))

#         if self.laplaceSmoothing:
#             self.T1 = (Nj1 + 1)/(Ny1 + 2)
#             self.T0 = (Nj0 + 1)/(Ny0 + 2)
#         else:
#             self.T1 = Nj1/Ny1
#             self.T0 = Nj0/Ny0

#         return self

#     def predict(self, X):

#         ## ensure that the model has been trained before predicting
#         try:
#             getattr(self, "T0")
#             getattr(self, "T1")
#             getattr(self, "theta0")
#             getattr(self, "theta1")
#         except AttributeError:
#             raise RuntimeError("You must train classifer before predicting data!")

#         ## calculate probablity function, delta (lecture 9, slide 5)
#         delta = (  np.log(self.theta1/(1-self.theta1))
#                  + X.dot(np.log(self.T1/self.T0))
#                  + (1-X.todense()).dot(np.log((1-self.T1)/(1-self.T0))) )


#         ## make prediction from learned weights
#         pred = np.zeros(delta.shape).astype('int')
#         pred[delta > 0] = 1

#         return pred.reshape([-1,])

#     def score(self, X, y):
#         ## ensure features and target are binary
#         assert np.array_equal(y, y.astype(bool))

#         ## predict output from learned weights
#         pred = self.predict(X)

#         ## compare to true targets
#         diff = np.equal(pred, y)

#         ## sum how many examples are correctly predicted
#         score = np.sum(diff)/y.shape[0]
#         return score
