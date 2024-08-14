import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse


# 计算基于BERT的相似度
def calculate_bert_similarity(hist_features, target_feature, model):
    hist_features_vec = model.encode(hist_features, convert_to_tensor=True)
    target_feature_vec = model.encode([target_feature], convert_to_tensor=True)
    content_sim_matrix = cosine_similarity(hist_features_vec.cpu().numpy(), target_feature_vec.cpu().numpy()).flatten()
    return content_sim_matrix


# 主函数
parser = argparse.ArgumentParser()
parser.add_argument("--pooling", type=str, default="average")
parser.add_argument("--alpha", type=float, default=0.5)
args = parser.parse_args()

# 加载数据
isbn2id = json.load(open(f"data/book/isbn2id.json"))
id2book = json.load(open(f"data/book/id2book.json"))
embeddings = np.load(f"./embeddings/BookCrossing_{args.pooling}.npy")
normalized_embeddings = normalize(embeddings)
print("Embeddings loaded.")
print(embeddings.shape)

all_book_features = {}
for i in range(len(id2book)):
    isbn, title, author, year, publisher = id2book[str(i)]
    text = \
        f"Here is a book. Its title is {title}. ISBN of the book is {isbn}. The author of the book is {author}. "\
        f"The publication year of the book is {year}. Its publisher is {publisher}."
    all_book_features[str(i)] = text

# 加载预训练的BERT模型
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # 可以选择其他预训练模型

set_list = ['train', 'valid', 'test']
# 处理数据集
for set in set_list:
    path = f"data/book/{set}.csv"
    print(set)
    df = pd.read_csv(f"data/book/{set}.csv")
    df = df.reset_index(drop=True)
    all_indice = []
    for idx, row in tqdm(df.iterrows()):
        tgt_id = int(row['item_id'])
        hist_id = [int(i) for i in eval(row['history_item_id'])]

        tgt_embed_norm = normalized_embeddings[tgt_id].reshape(1, -1)
        hist_embed_norm = normalized_embeddings[hist_id]

        # 计算基于嵌入的相似度
        embed_cos_sim_matrix = cosine_similarity(hist_embed_norm, tgt_embed_norm).flatten()

        # 提取特征信息
        target_feature = all_book_features[str(tgt_id)]
        hist_features = [all_book_features[str(i)] for i in hist_id]

        # 计算基于BERT的相似度
        content_sim_matrix = calculate_bert_similarity(hist_features, target_feature, bert_model)

        # 结合两种相似度
        combined_similarity = args.alpha * embed_cos_sim_matrix + (1 - args.alpha) * content_sim_matrix

        seq_id_to_book_id = {i: book_id for i, book_id in enumerate(hist_id)}
        indice = np.argsort(-combined_similarity)[:100].tolist()
        sorted_indice = list(map(lambda x: int(seq_id_to_book_id[x]), indice))
        all_indice.append(sorted_indice)

    json.dump(all_indice, open(f'./embeddings/BookCrossing_{args.pooling}_indice_{set}.json', 'w'), indent=4)
