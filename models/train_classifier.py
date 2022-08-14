from copyreg import pickle
import sys
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re 
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """Loads a dataset from a given databse

    Args:
        database_filepath (filepath): filepath for the database that contains the data

    Returns:
        Array: Returns the features, the target and the category names for the targets
    """
    # creates sqlite engine 
    engine = create_engine(f'sqlite:///{database_filepath}')

    # reads table from database 
    df = pd.read_sql_table('messages', engine)

    # gets the feature and the target variables 
    X = df.message
    y = df.iloc[:, 4:]

    # gets the category names 
    category_names = y.columns.to_list()

    return X, y, category_names


def tokenize(text):
    """Tokenizes a given text

    Args:
        text (str): text that will be tokenized

    Returns:
        list: list of words after tokenization 
    """
    # make all text lowercase 
    text = text.lower()

    # remove all type of punctuations 
    text = re.sub(f'[^a-z0-9]',' ', text)

    # tokenize the text
    tokens = word_tokenize(text)

    # lemmatize the text
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    # remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    return tokens



def build_model():
    """Creates a machine learning pipeline for multiclass classification 

    Returns:
        obj: returns the machine learning model that will be used to classify the messages 
    """
    # creates pipeline to apply transformers and model 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # parameters grid for GridSearch
    params = {
        'tfidf__use_idf': (True, False),
        'moc__estimator__n_estimators': [50, 60, 70]
    }

    # instantiates model 
    model = GridSearchCV(pipeline, param_grid=params)

    return model 



def evaluate_model(model, X_test, Y_test):
    """Trains and evaluates a machine learning model with test data

    Args:
        model (obj): the machine learning model that will be trained 
        X_test (array): array of testing data 
        Y_test (array): array of the correct testing results 
    """
    # predicts test data using model 
    Y_pred = model.predict(X_test)
    
    # iterates through columns to show results in each category
    i = 0
    for col in Y_test:
        # prints confusion matrix
        plt.title(f'Feature: {col}')
        sns.heatmap(confusion_matrix(Y_test[col], Y_pred[:,i]))
        plt.show()
        # prints classification report
        print(classification_report(Y_test[col], Y_pred[:,i], zero_division=0))
        i+=1
    # computes model's general accuracy 
    accuracy = (Y_test.values == Y_pred).mean()
    print(f'accuracy: {accuracy}')



def save_model(model, model_filepath):
    """Saves a machien learning model for reusing 

    Args:
        model (obj): the model that will be saved
        model_filepath (str): the filepath for the model's file 
    """
    # opens pickle file to save the model 
    with open(model_filepath) as file:
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