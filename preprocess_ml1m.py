import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

f = open('raw_data/movie/ml-1m/ratings.dat', 'r')
data = f.readlines()
f = open('raw_data/movie/ml-1m/movies.dat', 'r', encoding='ISO-8859-1')
movies = f.readlines()
f = open('raw_data/movie/ml-1m/users.dat', 'r')
users = f.readlines()

movie_names = [_.strip().split("::")[1] for _ in movies]  # movie_names[0] = 'Toy Story (1995)'
user_ids = [_.strip().split("::")[0] for _ in users]  # user_ids[0] = '1'
movie_ids = [_.strip().split("::")[0] for _ in movies]  # movie_ids[0] = '1'
movie_genre = [_.strip().split("::")[-1] for _ in movies]

interaction_dicts = dict()
id2name = dict()
for i, j, k in zip(movie_ids, movie_names, movie_genre):
    id2name[i] = [j, k]

json.dump(id2name, open(os.path.join('data/movie/ml-1m', "movie_detail.json"), "w"))


for line in data:
    user_id, movie_id, rating, timestamp = line.strip().split('::')
    if user_id not in interaction_dicts:
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
        }
    interaction_dicts[user_id]['movie_id'].append(movie_id)
    interaction_dicts[user_id]['rating'].append(int(rating))
    interaction_dicts[user_id]['timestamp'].append(timestamp)

seq_len = args.len
user_under_10 = []
for user in user_ids:
    if len(interaction_dicts[user]['movie_id']) < seq_len:
        user_under_10.append(user)


json.dump(interaction_dicts, open(os.path.join('data/movie/ml-1m', "interaction_dicts.json"), "w"))


sequential_interaction_list = []
cold_start_interaction_list = []
for user_id in interaction_dicts:
    if user_id in user_under_10:
        cold_start_interaction_list.append(
            [user_id, interaction_dicts[user_id]['movie_id'][:-1], interaction_dicts[user_id]['rating'][:-1],
             interaction_dicts[user_id]['movie_id'][-1], interaction_dicts[user_id]['rating'][-1],
             interaction_dicts[user_id]['timestamp'][-1].strip('\n')]
        )
    temp = zip(interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'],
               interaction_dicts[user_id]['timestamp'])
    temp = sorted(temp, key=lambda x: x[2])
    result = zip(*temp)
    interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id][
        'timestamp'] = [list(_) for _ in result]
    for i in range(seq_len, len(interaction_dicts[user_id]['movie_id'])):
        sequential_interaction_list.append(
            [user_id, interaction_dicts[user_id]['movie_id'][i - seq_len:i],
             interaction_dicts[user_id]['rating'][i - seq_len:i], interaction_dicts[user_id]['movie_id'][i],
             interaction_dicts[user_id]['rating'][i], interaction_dicts[user_id]['timestamp'][i].strip('\n')]
        )

import random
random.seed(args.seed)
print(len(sequential_interaction_list))
sample = 10000
random_sample = [random.randint(0, len(sequential_interaction_list)) for _ in range(sample)]
sequential_interaction_list = [sequential_interaction_list[i] for i in random_sample]

import csv

# save the csv file for baselines
with open('./data/movie/ml-1m/train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[:int(len(sequential_interaction_list) * 0.8)])
with open('./data/movie/ml-1m/valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[
                     int(len(sequential_interaction_list) * 0.8):int(len(sequential_interaction_list) * 0.9)])
with open('./data/movie/ml-1m/test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list) * 0.9):])

with open('./data/movie/ml-1m/cold_start_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(cold_start_interaction_list)