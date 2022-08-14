import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
#from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # visual 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # visual 2 
    categories_sum = df.iloc[:, 5:].sum().sort_values(ascending=False)
    categories_name = categories_sum.index.tolist()

    # visual 3
    top_10_categories_sum = df.iloc[:, 5:].sum().sort_values(ascending=False)[:10]
    top_10_categories_name = top_10_categories_sum.index.tolist()

    # visual 4
    botton_10_categories_sum = df.iloc[:, 5:].sum().sort_values(ascending=False)[-10:-1]
    botton_10_categories_name = botton_10_categories_sum.index.tolist()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = categories_name,
                    y = categories_sum
                )
            ],

            'layout': {
                'tilte': 'Amount of Messages for Each Category',
                'yaxis': {
                    'title': 'Amount of Messages'
                },
                'xaxis': {
                    'title': 'Categories'
                }
            }
        },
        {
            'data': [
                Bar(
                    x = top_10_categories_name,
                    y = top_10_categories_sum
                )
            ],

            'layout': {
                'title': 'Top 10 Most Commom Messages Categories',
                'yaxis': {
                    'title': 'Amount of Messages'
                },
                'xaxis': {
                    'title': 'Categories'
                }
            }
        },
        {
            'data': [ 
                Bar(
                    x = botton_10_categories_name,
                    y = botton_10_categories_sum
                )
            ],

            'layout': {
                'title': 'Least Commom Messages Categories',
                'yaxis': {
                    'title': 'Amount of Messages'
                },
                'xaxis': {
                    'title': 'Categories'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()