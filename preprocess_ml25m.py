import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

age_dict = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "above 56"
}

job_dict = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

f = open('raw_data/movie/ml-25m/ratings.csv', 'r')
data = f.readlines()[1:]
f = open('raw_data/movie/ml-25m/movies.csv', 'r', encoding='ISO-8859-1')
movies = f.readlines()[1:]
movie_names = []
movie_ids = []
movie_genre = []

new_movie_id_dict = dict()
new_movie_id = 1
for line in tqdm(movies):
    row = line.strip().split(',')
    new_movie_id_dict[row[0]] = str(new_movie_id)
    new_movie_id += 1


for line in tqdm(movies):
    row = line.strip().split(',')
    movie_ids.append(new_movie_id_dict[row[0]])
    movie_names.append(row[1])
    movie_genre.append(row[2])


id2name = dict()
for i, j, k in zip(movie_ids, movie_names, movie_genre):
    id2name[i] = [j, k]
print(id2name[movie_ids[0]])
print(len(id2name))

json.dump(id2name, open(os.path.join('data/movie/ml-25m', "movie_detail.json"), "w"))

interaction_dicts = dict()

user_ids = []
for line in tqdm(data):
    row = line.strip().split(',')
    user_id, movie_id, rating, timestamp = row
    if user_id not in interaction_dicts:
        user_ids.append(user_id)
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
        }
    interaction_dicts[user_id]['movie_id'].append(new_movie_id_dict[movie_id])
    interaction_dicts[user_id]['rating'].append(rating)
    interaction_dicts[user_id]['timestamp'].append(timestamp)

json.dump(interaction_dicts, open(os.path.join('data/movie/ml-25m', "interaction_dicts.json"), "w"))

seq_len = args.len
user_under_10 = []
for user in user_ids:
    if len(interaction_dicts[user]['movie_id']) < seq_len:
        user_under_10.append(user)

sequential_interaction_list = []
cold_start_interaction_list = []
for user_id in tqdm(interaction_dicts):
    if user_id in user_under_10:
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
with open('./data/movie/ml-25m/train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[:int(len(sequential_interaction_list) * 0.8)])
with open('./data/movie/ml-25m/valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[
                     int(len(sequential_interaction_list) * 0.8):int(len(sequential_interaction_list) * 0.9)])
with open('./data/movie/ml-25m/test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list) * 0.9):])

with open('./data/movie/ml-1m/cold_start_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(cold_start_interaction_list)