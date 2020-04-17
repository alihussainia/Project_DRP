
# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle

from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score

def load_data(database_filepath):
    filename = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(filename)
    df = pd.read_sql('Messages',engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)
    category = Y.columns.values
    return X, Y, category

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-1]", " ", text)
    
    #Tokenize
    tokens = word_tokenize(text)
    #Stopword
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():

    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier(),n_jobs=-1))
                ])
    
    #Lesser parameters given to save processing time
    parameters =  {'vect__ngram_range': ((1, 1), (1, 2)),
                   #'tfidf__use_idf': (True, False),
                   #'clf__estimator__min_samples_split': [2, 3, 4]
                   }
    #Pipeline is passed on to the gridsearch
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates through classification report
    Args:   Model, features, labels and category list
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    for  idx, category in enumerate(Y_test.columns.values):
        print("{} -- {}".format(category, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
    print("Accuracy of the model: {}".format(accuracy_score(Y_test, y_pred)))


def save_model(model, model_filepath):
    '''
    Save as pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

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