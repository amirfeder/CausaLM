"""Create train/dev/test data with and without adjectives"""
from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_DOMAINS
from datasets.datasets_utils import output_datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import xml.etree.ElementTree as ET

## Model Hyperparams
num_topics = 30
num_features = 500
num_top_words = 10

def XML2arrayRAW(cur_path):
    reviews = []
    tree = ET.parse(cur_path)
    root = tree.getroot()
    for rev in root.iter('review'):
        reviews.append(rev.text)
    return reviews


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\n Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def get_topic_distribution(data):
    tf_vectorizer = CountVectorizer(max_features=num_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(data)
    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',
                                    learning_offset=50., random_state=0).fit(tf)

    # display_topics(lda, tf_feature_names, num_top_words)
    return lda.transform(tf)



for domain in SENTIMENT_DOMAINS:
    domain_data = []
    domain_labels = []
    examples = []
    DOMAIN_DIR = os.path.join(SENTIMENT_RAW_DATA_DIR, domain)

    for key in output_datasets.keys():
        DATA_FILE = os.path.join(DOMAIN_DIR, output_datasets[key] + '.parsed')
        cur_data = XML2arrayRAW(DATA_FILE)

        domain_data += cur_data
        domain_labels += [key] * len(cur_data)

    domain_topic_dist = get_topic_distribution(domain_data)
    domain_data = list(zip(domain_labels, domain_data))

    topics_df = pd.DataFrame(np.array(domain_topic_dist), columns=['topic_'+str(i+1) for i in range(num_topics)])
    review_df = pd.DataFrame(np.array(domain_data), columns=['label', 'review'])
    df = pd.concat([review_df, topics_df], axis=1)

    for i in range(num_topics):
        topic_average = df['topic_' + str(i+1)].mean()
        df['topic_bin_' + str(i + 1)] = (df['topic_' + str(i + 1)] > topic_average).astype(int)

    train, test = train_test_split(df, test_size=0.2)
    train, dev = train_test_split(train, test_size=0.2)
    train.to_csv(SENTIMENT_RAW_DATA_DIR + '/' + domain + '/' + 'topics_train.csv')
    dev.to_csv(SENTIMENT_RAW_DATA_DIR + '/' + domain + '/' + 'topics_dev.csv')
    test.to_csv(SENTIMENT_RAW_DATA_DIR + '/' + domain + '/' + 'topics_test.csv')

    print('Done with: ' + domain)

