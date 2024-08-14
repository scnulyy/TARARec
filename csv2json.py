import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="book")
parser.add_argument("--len", type=int, default=10)
parser.add_argument("--relevant_num", type=int, default=0)
parser.add_argument("--recent_num", type=int, default=10)
parser.add_argument("--type", type=str, default='zero-shot')
args = parser.parse_args()

if args.dataset == 'book':
    id2name = json.load(open(f'data/book/id2book.json'))
    isbn2id = json.load(open(f'data/book/isbn2id.json'))
    interaction_dir = f'data/book/interaction_dicts.json'
else:
    id2name = json.load(open(f'data/movie/{args.dataset}/movie_detail.json'))
    interaction_dir = f'data/movie/{args.dataset}/interaction_dicts.json'
interaction_dicts = json.load((open(interaction_dir)))


def csv_to_json_ml1m(input_path, output_path):
    sorted_indice = np.load(f'embeddings/ml-1m_average_indice.npy')
    data = pd.read_csv(input_path)
    json_list = []
    print("relevant: ", args.relevant_num)
    print("recent_num: ", args.recent_num)
    print("len: ", args.len)
    # ['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp']
    for index, row in tqdm(data.iterrows()):
        row['history_movie_id'] = eval(row['history_movie_id'])
        row['history_rating'] = eval(row['history_rating'])
        target_movie_id = row['movie_id']
        user = str(row['user_id'])
        user_full_interactions = interaction_dicts[user]['movie_id']
        user_full_rating = interaction_dicts[user]['rating']
        recent_interactions, recent_rating = row['history_movie_id'], row['history_rating']
        relevant_interactions, relevant_rating = [], []

        if args.relevant_num > 0:
            cur_indice = sorted_indice[target_movie_id - 1, :]
            cnt = 0
            for index in cur_indice:
                if str(index) in user_full_interactions and str(index) != target_movie_id:
                    cnt += 1
                    relevant_interactions.append(index)
                    rating_index = user_full_interactions.index(str(index))
                    relevant_rating.append(user_full_rating[rating_index])
                    if cnt == args.relevant_num:
                        break

        relevant = []
        recent = []
        relevant_str = ""
        recent_str = ""

        # 相似
        for i in range(args.relevant_num):
            name_rating_str = str(id2name[str(relevant_interactions[i])][0]) + " , rating: " + str(relevant_rating[i])
            relevant.append(name_rating_str)
        # 近期
        for i in range(len(recent_interactions)):
            name_rating_str = str(id2name[str(recent_interactions[i])][0]) + " , rating: " + str(recent_rating[i])
            recent.append(name_rating_str)

        target_movie = id2name[str(target_movie_id)][0]

        for i in range(args.relevant_num):
            relevant_str += "\n\"" + relevant[i] + "\""

        for i in range(args.recent_num):
            recent_str += "\n\"" + recent[i] + "\""

        target_preference = int(row['rating'])
        target_movie_str = "\"" + target_movie + "\""
        target_preference_str = "Yes." if target_preference > 3 else "No."
        if args.recent_num > 0 and args.relevant_num > 0:
            json_list.append({
                "instruction": "Given the user interaction history (including items related to the target item that the user has interacted with and recent items that the user has interacted with) and ratings by the user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User relevant interactions: {relevant_str}\nUser recent interactions: {recent_str}\nWhether the user will like the target item {target_movie_str}?",
                "output": target_preference_str,
            })
        elif args.recent_num == 0 and args.relevant_num == args.len:
            json_list.append({
                "instruction": "Given the user interaction history and ratings by user (0-10), identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {relevant_str}\nWhether the user will like the target item {target_movie_str}?",
                "output": target_preference_str,
            })
        elif args.relevant_num == 0 and args.recent_num == args.len:
            json_list.append({
                "instruction": "Given the user interaction history and ratings by user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {recent_str}\nWhether the user will like the target item {target_movie_str}?",
                "output": target_preference_str,
            })
    print('sample: ', len(json_list))
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


def csv_to_json_ml25m(input_path, output_path, mode):
    indice_dir = f'./embeddings/ml-25m_average_indice_{mode}.json'
    sorted_indice = json.load(open(indice_dir))
    relevant_num = args.relevant_num
    recent_num = args.recent_num
    data = pd.read_csv(input_path)
    json_list = []
    # ['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp']
    for index, row in tqdm(data.iterrows()):
        item_id = str(row['movie_id'])
        target_movie = id2name[item_id][0]
        row['history_movie_id'] = eval(row['history_movie_id'])
        row['history_rating'] = eval(row['history_rating'])
        user = str(row['user_id'])
        user_full_interactions = interaction_dicts[user]['movie_id']
        user_full_rating = interaction_dicts[user]['rating']
        recent_interactions, recent_rating = row['history_movie_id'], row['history_rating']
        relevant_interactions, relevant_rating = [], []
        cur_indice = sorted_indice[index]
        if len(cur_indice) < args.len:
            continue
        if len(recent_interactions) < recent_num:
            continue
        for i in range(relevant_num):
            relevant_interactions.append(str(cur_indice[i]))
            rating_index = user_full_interactions.index(str(cur_indice[i]))
            relevant_rating.append(user_full_rating[rating_index])

        interaction = []
        relevant = []
        recent = []
        relevant_str = ""
        recent_str = ""

        # 相似
        for i in range(args.relevant_num):
            name_rating_str = str(id2name[str(relevant_interactions[i])][0]) + " , rating: " + str(relevant_rating[i])
            interaction.append(str(relevant_interactions[i]))
            relevant.append(name_rating_str)
        # 近期
        if args.recent_num > 0:
            for i in range(len(recent_interactions)):
                if str(recent_interactions[i]) not in interaction:
                    name_rating_str = str(id2name[str(recent_interactions[i])][0]) + " , rating: " + str(recent_rating[i])
                    recent.append(name_rating_str)
                    interaction.append(str(recent_interactions[i]))
                if len(interaction) == args.len:
                    break

        target_preference = int(row['rating'])
        target_movie_str = "\"" + target_movie + "\""
        target_preference_str = "Yes." if target_preference > 3 else "No."

        for i in range(relevant_num):
            relevant_str += "\n\"" + relevant[i] + "\""

        for i in range(recent_num):
            recent_str += "\n\"" + recent[i] + "\""

        if args.recent_num > 0 and args.relevant_num > 0:
            json_list.append({
                "instruction": "Given the user interaction history (including items related to the target item that the user has interacted with and recent items that the user has interacted with) and ratings by the user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User relevant interactions: {relevant_str}\nUser recent interactions: {recent_str}\nWhether the user will like the target item {target_movie_str}?",
                "output": target_preference_str,
            })
        elif args.recent_num == 0 and args.relevant_num == args.len:
            json_list.append({
                "instruction": "Given the user interaction history and ratings by user (0-10), identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {relevant_str}\nWhether the user will like the target item {target_movie_str}?",
                "output": target_preference_str,
            })
        elif args.relevant_num == 0 and args.recent_num == args.len:
            json_list.append({
                "instruction": "Given the user interaction history and ratings by user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {recent_str}\nWhether the user will like the target item {target_movie_str}?",
                "output": target_preference_str,
            })
    print(len(json_list))
    print(json_list[0])
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


def csv_to_json_bc(input_path, output_path, mode):
    indice_dir = f'./embeddings/BookCrossing_average_indice_{mode}.json'
    sorted_indice = json.load(open(indice_dir))
    relevant_num = args.relevant_num
    recent_num = args.recent_num
    data = pd.read_csv(input_path)
    json_list = []
    # ['user', 'history_item_id', 'history_rating', 'item_id', 'rating']
    for index, row in tqdm(data.iterrows()):
        item_id = str(row['item_id'])
        target_book_title = id2name[item_id][1]
        target_book_rating = row['rating']
        row['history_item_id'] = eval(row['history_item_id'])
        row['history_rating'] = eval(row['history_rating'])
        user = str(row['user'])
        user_full_interactions = interaction_dicts[user]['ISBN']
        user_full_rating = interaction_dicts[user]['Book-Rating']
        recent_interactions, recent_rating = row['history_item_id'], row['history_rating']
        relevant_interactions, relevant_rating = [], []
        cur_indice = sorted_indice[index]
        if len(cur_indice) < args.len:
            continue
        if len(recent_interactions) < recent_num:
            continue
        for i in range(relevant_num):
            relevant_interactions.append(cur_indice[i])
            rating_index = user_full_interactions.index(cur_indice[i])
            relevant_rating.append(user_full_rating[rating_index])

        interaction = []
        relevant = []
        recent = []
        relevant_str = ""
        recent_str = ""

        # 相似
        for i in range(args.relevant_num):
            name_rating_str = str(id2name[str(relevant_interactions[i])][1]) + " , rating: " + str(relevant_rating[i])
            interaction.append(str(relevant_interactions[i]))
            relevant.append(name_rating_str)
        # 近期
        if args.recent_num > 0:
            for i in range(len(recent_interactions)):
                if str(recent_interactions[i]) not in interaction:
                    name_rating_str = str(id2name[str(recent_interactions[i])][1]) + " , rating: " + str(recent_rating[i])
                    recent.append(name_rating_str)
                    interaction.append(str(recent_interactions[i]))
                if len(interaction) == args.len:
                    break

        target_preference_str = "Yes." if target_book_rating > 5 else "No."
        target_book_str = "\"" + target_book_title + "\""

        for i in range(relevant_num):
            relevant_str += "\n\"" + relevant[i] + "\""

        for i in range(recent_num):
            recent_str += "\n\"" + recent[i] + "\""

        if args.recent_num > 0 and args.relevant_num > 0:
            json_list.append({
                "instruction": "Given the user interaction history (including items related to the target item that the user has interacted with and recent items that the user has interacted with) and ratings by the user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User relevant interactions: {relevant_str}\nUser recent interactions: {recent_str}\nWhether the user will like the target item {target_book_str}?",
                "output": target_preference_str,
            })
        elif args.recent_num == 0 and args.relevant_num == args.len:
            json_list.append({
                "instruction": "Given the user interaction history and ratings by user (0-10), identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {relevant_str}\nWhether the user will like the target item {target_book_str}?",
                "output": target_preference_str,
            })
        elif args.relevant_num == 0 and args.recent_num == args.len:
            json_list.append({
                "instruction": "Given the user interaction history and ratings by user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {recent_str}\nWhether the user will like the target item {target_book_str}?",
                "output": target_preference_str,
            })
    print(len(json_list))
    print(json_list[0])
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


def csv_to_json_bc_cold_start(input_path, output_path):
    data = pd.read_csv(input_path)
    json_list = []
    # ['user', 'history_item_id', 'history_rating', 'item_id', 'rating']
    for index, row in tqdm(data.iterrows()):
        item_id = str(row['item_id'])
        target_book_title = id2name[item_id][1]
        target_book_rating = row['rating']
        interactions = eval(row['history_item_id'])
        rating = eval(row['history_rating'])
        interaction_len = len(interactions)

        interaction = []
        interaction_str = ""

        for i in range(interaction_len):
            name_rating_str = str(id2name[str(interactions[i])][1]) + " , rating: " + str(rating[i])
            interaction.append(name_rating_str)

        target_preference_str = "Yes." if target_book_rating > 5 else "No."
        target_book_str = "\"" + target_book_title + "\""

        for i in range(interaction_len):
            interaction_str += "\n\"" + interaction[i] + "\""

        json_list.append({
            "instruction": "Given the user interaction history and ratings by user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
            "input": f"User History: {interaction_str}\nWhether the user will like the target item {target_book_str}?",
            "output": target_preference_str,
        })
    print(len(json_list))
    print(json_list[0])
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


def csv_to_json_ml_cold_start(input_path, output_path):
    data = pd.read_csv(input_path)
    json_list = []
    # ['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp']
    for index, row in tqdm(data.iterrows()):

        item_id = str(row['movie_id'])
        target_movie_title = id2name[item_id][0]
        target_movie_rating = row['rating']
        interactions = eval(row['history_movie_id'])
        rating = eval(row['history_rating'])
        interaction_len = len(interactions)

        interaction = []
        interaction_str = ""

        for i in range(interaction_len):
            name_rating_str = str(id2name[str(interactions[i])][0]) + " , rating: " + str(rating[i])
            interaction.append(name_rating_str)

        target_preference_str = "Yes." if target_movie_rating > 3 else "No."
        target_movie_str = "\"" + target_movie_title + "\""

        for i in range(interaction_len):
            interaction_str += "\n\"" + interaction[i] + "\""

        json_list.append({
            "instruction": "Given the user interaction history and ratings by user, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
            "input": f"User History: {interaction_str}\nWhether the user will like the target item {target_movie_str}?",
            "output": target_preference_str,
        })
    print(len(json_list))
    print(json_list[0])
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


if args.type == 'few-shot':
    if args.dataset == 'book':
        csv_to_json_bc('data/book/train.csv', 'data/book/train.json', mode='train')
        csv_to_json_bc('data/book/valid.csv', 'data/book/valid.json', mode='valid')
        csv_to_json_bc('data/book/test.csv', 'data/book/test.json', mode='test')

    elif args.dataset == 'ml-1m':
        csv_to_json_ml1m(f'./data/movie/ml-1m/train.csv', f'./data/movie/ml-1m/train.json')
        csv_to_json_ml1m(f'./data/movie/ml-1m/valid.csv', f'./data/movie/ml-1m/valid.json')
        csv_to_json_ml1m(f'./data/movie/ml-1m/test.csv', f'./data/movie/ml-1m/test.json')

    elif args.dataset == 'ml-25m':
        csv_to_json_ml25m(f'./data/movie/ml-25m/train.csv', f'./data/movie/ml-25m/train.json', mode='train')
        csv_to_json_ml25m(f'./data/movie/ml-25m/valid.csv', f'./data/movie/ml-25m/valid.json', mode='valid')
        csv_to_json_ml25m(f'./data/movie/ml-25m/test.csv', f'./data/movie/ml-25m/test.json', mode='test')

elif args.type == 'zero-shot':
    if args.dataset == 'book':
        csv_to_json_bc('data/book/test_zero_shot.csv', 'data/book/test_zero_shot.json', mode='test')

    else:
        csv_to_json_ml1m(f'./data/movie/{args.dataset}/test_zero_shot.csv', f'./data/movie/{args.dataset}/test_zero_shot.json')