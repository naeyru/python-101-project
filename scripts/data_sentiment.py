"""
Thomas Safago
11/10/2023
Sentiment analysis test/setup.
"""


import pandas as pd
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess_text(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def get_sentiment(text):
    scores = analyzer.polarity_scores(preprocess_text(text))
    return scores["compound"]  # Range of [-1, 1].


def data_sentiment(*args):
    for filepath in args:
        data = pd.read_csv("new_data/" + filepath, low_memory=False)

        text_column = data["text"]
        new_text_column = text_column.map(get_sentiment)
        data["text"] = new_text_column

        data.to_csv("new_data/sentiment_" + filepath, index=False)
        print(f"<{filepath}> done!")


def main():
    data_sentiment("BTCP1.csv",  "BTCP2.csv",  "BTCP3.csv",  "BTCP4.csv",  "BTCP5.csv",  "BTCP6.csv",  "BTCP7.csv",  "BTCP8.csv")
    # Note: A LOT of tweets will be rated as 0. These are largely bot tweets and MUST BE DISREGARDED when doing more stuff with this data.


if __name__ == "__main__":
    main()
