import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

class Models:

    def __init__(self):
        self.name = ''
        path = 'dataset/trainingdata.csv'
        df = pd.read_csv(path)
        df = df.dropna()
        self.x = df['sentences']
        self.y = df['sentiments']

    def mnb_classifier(self):
        self.name = 'MultinomialNB classifier'
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3)
        }
        classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', GridSearchCV(MultinomialNB(), parameters, cv=5)),
        ])
        return classifier.fit(self.x, self.y)

    def svm_classifier(self):
        self.name = 'SVM classifier'
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf-svm__alpha': (1e-2, 1e-3)
        }
        classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', GridSearchCV(SGDClassifier(loss='hinge', penalty='l2', random_state=42), parameters, cv=5)),
        ])
        classifier = classifier.fit(self.x, self.y)
        pickle.dump(classifier, open(self.name + '.pkl', "wb"))
        return classifier

    def mnb_stemmed_classifier(self):
        self.name = 'MultinomialNB stemmed classifier'
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'mnb__alpha': (1e-2, 1e-3)
        }
        self.stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
        classifier = Pipeline([
            ('vect', self.stemmed_count_vect),
            ('tfidf', TfidfTransformer()),
            ('mnb', GridSearchCV(MultinomialNB(fit_prior=False), parameters, cv=5)),
        ])
        classifier = classifier.fit(self.x, self.y)
        pickle.dump(classifier, open(self.name + '.pkl', "wb"))
        return classifier

    def svm_stemmed_classifier(self):
        self.name = 'SVM stemmed classifier'
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf-svm__alpha': (1e-2, 1e-3)
        }
        self.stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
        classifier = Pipeline([
            ('vect', self.stemmed_count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', GridSearchCV(SGDClassifier(), parameters, cv=5)),
        ])
        classifier = classifier.fit(self.x, self.y)
        pickle.dump(classifier, open(self.name + '.pkl', "wb"))
        return classifier

    def accuracy(self, model):
        predicted = model.predict(self.x)
        accuracy = np.mean(predicted == self.y)
        print(f"{self.name} has accuracy of {accuracy * 100} % ")

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


if __name__ == '__main__':
    model = Models()
    model.accuracy(model.mnb_classifier())
    model.accuracy(model.svm_classifier())
    model.accuracy(model.mnb_stemmed_classifier())
    model.accuracy(model.svm_stemmed_classifier())
