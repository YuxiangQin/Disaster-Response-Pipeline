import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

import numpy as np
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('data', engine.connect())
    df.dropna(subset=['related'], inplace=True)
    X = df['message']
    Y = df.iloc[:,-36:]
    return X, Y, Y.columns


def tokenize(text):
    text = text.lower()
    # remove punctuation characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # tokenize
    words = word_tokenize(text)
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatizing
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__max_depth': [None, 10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    reports = []
    precisions = []
    recalls = []
    f1_scores = []

    for i, category in enumerate(category_names):
        report = classification_report(y_test.iloc[:, i], y_pred[:, i])
        precision = precision_score(y_test.iloc[:, i], y_pred[:, i])
        recall = recall_score(y_test.iloc[:, i], y_pred[:, i])
        f1 = f1_score(y_test.iloc[:, i], y_pred[:, i])
        print(category)
        print(report)
        reports.append(report)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        # return reports, precisions, recalls, f1_scores

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model.best_estimator_, file) # only save best_estimator in GridSearchCV object


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
