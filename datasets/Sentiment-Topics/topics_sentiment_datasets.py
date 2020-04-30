"""Create train/dev/test data with and without adjectives"""
from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_DOMAINS, RANDOM_SEED, NUM_CPU
from datasets.datasets_utils import sentiment_output_datasets, split_data
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from Timer import timer


## LDA Model Hyperparams
num_topics = 30
num_features = 500
num_top_words = 10


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\n Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


@timer
def get_topic_distribution(data):
    tf_vectorizer = CountVectorizer(max_features=num_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(data)
    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',
                                    learning_offset=50., random_state=RANDOM_SEED, n_jobs=NUM_CPU).fit(tf)

    # display_topics(lda, tf_feature_names, num_top_words)
    return lda.transform(tf)


@timer
def main():
    for domain in SENTIMENT_DOMAINS:
        print('Started: ' + domain)
        domain_reviews = []
        domain_labels = []

        for key, val in sentiment_output_datasets.items():
            with open(f"{SENTIMENT_RAW_DATA_DIR}/{domain}/{val}_clean.parsed") as datafile:
                cur_reviews = datafile.readlines()
                domain_reviews += cur_reviews
                domain_labels += [key] * len(cur_reviews)

        domain_topic_dist = get_topic_distribution(domain_reviews)
        domain_data = list(zip(domain_labels, domain_reviews))

        topics_df = pd.DataFrame(np.array(domain_topic_dist),
                                 columns=['topic_' + str(i + 1) for i in range(num_topics)])
        review_df = pd.DataFrame(np.array(domain_data), columns=['label', 'review'])
        df = pd.concat([review_df, topics_df], axis=1)

        for i in range(num_topics):
            topic_average = df['topic_' + str(i + 1)].mean()
            df['topic_bin_' + str(i + 1)] = (df['topic_' + str(i + 1)] > topic_average).astype(int)

        split_data(df, f"{SENTIMENT_RAW_DATA_DIR}/{domain}", "topics")

        print('Done with: ' + domain)


if __name__ == "__main__":
    main()
