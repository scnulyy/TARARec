import os, argparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from sentence_transformers import SentenceTransformer
import json
# import faiss


def item_sim(args):
    all_movie_features = {}
    movie_dict = json.load(open(os.path.join('data/movie/ml-1m', "movie_detail.json"), "r"))
    for i in trange(1, 3953):
        key = str(i)
        if key not in movie_dict.keys():
            title, genre = "", ""
        else:
            title, genre = movie_dict[key]
        text = \
            f"Here is a movie. Its title is {title}. The movie's genre is {genre}."
        all_movie_features[key] = text

    bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    content_emb = [all_movie_features[i] for i in all_movie_features.keys()]
    content_emb = bert_model.encode(content_emb)

    fp = f"./embeddings/ml-1m_average.npy"
    embed = np.load(fp)
    embed = normalize(embed)

    sim_matrix = cosine_similarity(embed)
    content_sim_matrix = cosine_similarity(content_emb)
    sim_matrix = args.alpha * sim_matrix + (1 - args.alpha) * content_sim_matrix
    print("Similarity matrix computed.")
    print(sim_matrix.shape)

    sorted_indice = np.argsort(-sim_matrix, axis=1)
    print(sorted_indice.shape)
    print("Sorted.")

    fp_indice =os.path.join(args.embed_dir, '_'.join(["ml-1m", "average", "indice"])+".npy")
    np.save(fp_indice, sorted_indice)
    print("Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="./embeddings")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()
    item_sim(args)

