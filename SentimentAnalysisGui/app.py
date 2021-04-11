from flask import *
import numpy as np
import pandas as pd
import re
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


data = pd.read_csv('D:\\PycharmProject\\SentimentAnalysisGui\\static\\TweetsText.csv')
data = data[data['airline_sentiment'] != 'neutral']


def clean_tweet(text):
    tweet = ''
    tweet = re.sub('[' + string.punctuation + ']', '', text)
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', text)
    return tweet

data['text'] = data['text'].apply(lambda x: clean_tweet(x))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', '')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
pad_sequences(X)

baslik = ("Cümle", "Duygu Tahmini")

sentenceList = []
sentimentList = []
@app.route('/', methods=['GET', 'POST'])
def sentiment_analyisis():
    if request.method == 'POST':
        sentence = request.form['sentimentText']
        sentenceList.clear()
        sentimentList.clear()
        sentenceList.append(sentence)
        result(sentence)

    return render_template('index.html', headings=baslik, k=sentenceList, l=sentimentList)


def result(text):
    try:

        if text == '':
            flash("Try Again...", "warning")
        else:
            tweet = [text]
            tweet = tokenizer.texts_to_sequences(tweet)
            tweet = pad_sequences(tweet, maxlen=29, dtype='int32', value=0)
            model = load_model('D:\\PycharmProject\\SentimentAnalysisGui\\static\\best_model.h5')
            sentiment = model.predict(tweet, batch_size=1, verbose=0)[0]
            if (np.argmax(sentiment) == 0):
                # print("negative")
                flash("Negative Sentence", "danger")
                sentimentList.append('Negative Sentence')

            elif (np.argmax(sentiment) == 1):
                # print("positive")
                flash("Positive Sentence", "success")
                sentimentList.append('Positive Sentence')

    except BaseException as ex:
        print(ex)


if __name__ == '__main__':
    app.run(debug=True)
