import pandas as pd
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--type", type=str, default='few-shot')
args = parser.parse_args()

rating = pd.read_csv('raw_data/book/BX-Book-Ratings.csv', sep=';', encoding="latin-1")
users = pd.read_csv('raw_data/book/BX-Users.csv', sep=';', encoding="latin-1")
books = pd.read_csv('raw_data/book/BX-Books.csv', sep=';', encoding="latin-1", on_bad_lines="skip")
rating = pd.merge(rating, books, on='ISBN', how='inner')
books.to_csv('raw_data/book/book_item_mapping.csv', index=True)
print(len(books))
id2book = dict()

from tqdm import tqdm
user_dict = {}
item_id = {}
for index, row in tqdm(books.iterrows()):
    item_id[row['ISBN']] = index
    id2book[str(index)] = [
        row['ISBN'],
        row['Book-Title'],
        row['Book-Author'],
        row['Year-Of-Publication'],
        row['Publisher']
    ]
json.dump(id2book, open(os.path.join('data/book', "id2book.json"), "w"))
json.dump(item_id, open(os.path.join('data/book', "isbn2id.json"), "w"))

for index, row in tqdm(rating.iterrows()):
    userid = row['User-ID']
    if not user_dict.__contains__(userid):
        user_dict[userid] = {
            'ISBN': [],
            'Book-Rating': [],
            'Book-Title': [],
            'Book-Author': [],
            'Year-Of-Publication': [],
        }
    user_dict[userid]['ISBN'].append(item_id[row['ISBN']])
    user_dict[userid]['Book-Rating'].append(float(row['Book-Rating']))
    user_dict[userid]['Book-Title'].append(row['Book-Title'])
    user_dict[userid]['Book-Author'].append(row['Book-Author'])
    user_dict[userid]['Year-Of-Publication'].append(row['Year-Of-Publication'])

json.dump(user_dict, open(os.path.join('data/book', "interaction_dicts.json"), "w"))
new_user_dict = {}
cold_start_user = []
for key in user_dict.keys():

    if len(user_dict[key]['ISBN']) < 10:
        cold_start_user.append(key)
        continue
    elif max(user_dict[key]['Book-Rating']) <= 5:
        continue
    else:
        new_user_dict[key] = user_dict[key]

import random
import csv
random.seed(args.seed)
user_list = list(new_user_dict.keys())
random.shuffle(user_list)

with open('./data/book/cold_start_test.csv', 'w') as f:
    nrows = []
    for user in cold_start_user:
        item_id = user_dict[user]['ISBN']
        rating = user_dict[user]['Book-Rating']
        nrows.append([user, item_id[:-1], rating[:-1], item_id[-1], rating[-1]])
    writer = csv.writer(f)
    writer.writerow(['user', 'history_item_id', 'history_rating', 'item_id', 'rating'])
    writer.writerows(nrows)


if args.type == 'few-shot':
    train_user = user_list[:int(len(user_list) * 0.8)]
    valid_user = user_list[int(len(user_list) * 0.8):int(len(user_list) * 0.9)]
    test_user = user_list[int(len(user_list) * 0.9):]

    with open('./data/book/train.csv', 'w') as f:
        nrows = []
        for user in train_user:
            item_id = new_user_dict[user]['ISBN']
            rating = new_user_dict[user]['Book-Rating']
            nrows.append([user, item_id[:-1], rating[:-1], item_id[-1], rating[-1]])
        writer = csv.writer(f)
        writer.writerow(['user', 'history_item_id', 'history_rating', 'item_id', 'rating'])
        writer.writerows(nrows)

    with open('./data/book/valid.csv', 'w') as f:
        nrows = []
        for user in valid_user:
            item_id = new_user_dict[user]['ISBN']
            rating = new_user_dict[user]['Book-Rating']
            nrows.append([user, item_id[:-1], rating[:-1], item_id[-1], rating[-1]])
        writer = csv.writer(f)
        writer.writerow(['user', 'history_item_id', 'history_rating', 'item_id', 'rating'])
        writer.writerows(nrows)

    with open('./data/book/test.csv', 'w') as f:
        nrows = []
        for user in test_user:
            item_id = new_user_dict[user]['ISBN']
            rating = new_user_dict[user]['Book-Rating']
            nrows.append([user, item_id[:-1], rating[:-1], item_id[-1], rating[-1]])
        writer = csv.writer(f)
        writer.writerow(['user', 'history_item_id', 'history_rating', 'item_id', 'rating'])
        writer.writerows(nrows)


elif args.type == 'zero-shot':
    with open('./data/book/test_zero_shot.csv', 'w') as f:
        nrows = []
        for user in user_list:
            item_id = new_user_dict[user]['ISBN']
            rating = new_user_dict[user]['Book-Rating']
            nrows.append([user, item_id[:-1], rating[:-1], item_id[-1], rating[-1]])
        writer = csv.writer(f)
        writer.writerow(['user', 'history_item_id', 'history_rating', 'item_id', 'rating'])
        writer.writerows(nrows)
