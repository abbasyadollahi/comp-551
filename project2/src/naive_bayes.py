import numpy as np

class NaiveBayes:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, x, y):
        self.X = self.vectorizer.fit_transform(x).toarray()
        self.y = np.array(y)
        self.words = self.vectorizer.get_feature_names()

        self.prob_neg = (self.y == 0).sum() / self.y.size
        self.prob_pos = (self.y == 1).sum() / self.y.size

        self.mean_vector_neg = self.X[self.y == 0].mean(0)
        self.mean_vector_pos = self.X[self.y == 1].mean(0)

        self.var_vector_neg = self.X[self.y == 0].var(0)
        self.var_vector_pos = self.X[self.y == 1].var(0)

        self.neg_occurences = {0: (self.X[self.y == 0] == 0).sum(0), 1: (self.X[self.y == 0] == 1).sum(0)}
        self.pos_occurences = {0: (self.X[self.y == 1] == 0).sum(0), 1: (self.X[self.y == 1] == 1).sum(0)}

    def compute_feature_pdf(self, ft, mean, var):
        return np.exp(-(ft-mean)**2 / (2*var)) / np.sqrt(2 * np.pi * var) if var else 1

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

    def build_test_feature_matrix(self, x):
        return np.array([[1 if word in review else 0 for word in self.words] for review in x])

    def score(self, x, y):
        self.test_size = len(y)
        predictions = [self.predict(dp, False) for dp in self.build_test_feature_matrix(x)]
        return sum([1 if p == r else 0 for p, r in zip(predictions, y)]) / self.test_size
