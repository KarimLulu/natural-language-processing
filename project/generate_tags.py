import pandas as pd
import os
import numpy as np
import pickle
from utils import RESOURCE_PATH, question_to_vec

def load_embeddings(embeddings_path):
    embeddings = {}
    with open(embeddings_path) as f:
        for line in f:
            line = line.split("\t")
            embeddings[line[0]] = np.array([np.float(el) for el in line[1:]])
        dim = len(line[1:])
    return embeddings, dim

def main():
    starspace_embeddings, embeddings_dim = load_embeddings('word_embeddings.tsv')
    posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')
    counts_by_tag = posts_df.groupby("tag")["post_id"].count()
    os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

    for tag, count in counts_by_tag.items():
        tag_posts = posts_df[posts_df['tag'] == tag]

        tag_post_ids = tag_posts["post_id"].tolist()

        tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
        for i, title in enumerate(tag_posts['title']):
            tag_vectors[i, :] = question_to_vec(title, starspace_embeddings, embeddings_dim)

        # Dump post ids and vectors to a file.
        filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
        pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))

if __name__ == "__main__":
    main()
