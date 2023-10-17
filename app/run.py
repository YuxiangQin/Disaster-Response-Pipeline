import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
# from sklearn.externals import joblib
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
df = pd.read_sql_table('data', engine.connect())

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    df.dropna(subset=['related'], inplace=True)
    Y = df.iloc[:,-36:]
    viz = Y.sum().sort_values().reset_index()
    viz.columns = ['response_category', 'response_count']
    viz['response_category'] = viz.response_category.str.replace("_", " ").str.title() # format to camel
    viz['response_percentage'] = round(viz['response_count'] / viz['response_count'].sum() * 100, 1)
    hover_text = [f'{percentage}%' for percentage in viz.response_percentage]

    data = list(Y.sum(axis=1))

    # create visuals
    graphs = []

    # graph1: bar plot of category count
    graph_one =  {
        'data': [
            Bar(
                x=viz.response_count,
                y=viz.response_category,
                orientation='h',
                text=hover_text,
                hoverinfo='x + y + text'
            )
        ],

        'layout': {
            'title': 'Response Data by Category',
            'xaxis': {
                'title': "Response Count"
            },
            'margin': {
                'l': 100,
                't': 80,
                'b': 80
            },
            'width': 1200,
            'height': 800
            }
    }



    graphs.append(graph_one)
    

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
