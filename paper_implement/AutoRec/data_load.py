import numpy as np 
import pandas as pd
import _pickle as pickle
from sklearn.model_selection import train_test_split

def batch_iter(dataset, batch_size):
    """ make batch dataset for learning 
        Args :
            dataset : dataset to divide into batch segment
            batch_size : batch_size
        returns :
            dataset[start_idx:end_idx] : batch segments of dataset"""

    for idx in range(len(dataset) / batch_size):
        start_idx = idx * batch_size 
        end_idx = min((idx + 1) * batch_size, len(dataset)-1)
        input_mask = (dataset[start_idx:end_idx] != 0).astype(np.float32)
        yield list(zip(dataset[start_idx:end_idx], input_mask))

def load_data(file_path):
  """ load csv file and create dataset for deep learning
      Args :
        file_path : csv datafile path
      returns:
        dataset : shape(user_num, movie_num) matrix devided into train, test, dev set"""

  rating_data = pd.read_csv(file_path, sep=',', 
         names=['userId', 'movieId', 'rating', 'timestamp'])
  rating_data.rating = rating_data.rating.apply(pd.to_numeric, errors='coerce')

  indices = range(len(rating_data))
  train_indices, test_indices = train_test_split(indices, shuffle=True)
  train_indices, dev_indices = train_test_split(train_indices, test_size=0.1, shuffle=True)
  print("loading total data{} train {} dev {} test {}".format(
        len(indices), len(train_indices), len(dev_indices), len(test_indices)))

  num_users = rating_data.userId.nunique()
  num_movies = rating_data.movieId.nunique()
  print("users {} movies {}".format(num_users, num_movies))

  dataset = {
    'rating': np.zeros((num_users, num_movies), dtype=np.float32),
    'train' : {
        'mask' : np.zeros((num_users, num_movies), dtype=np.float32),
        'users' : set(), 
        'movies' : set()
      },
    'test' : {
        'mask' : np.zeros((num_users, num_movies), dtype=np.float32),
        'users' : set(),
        'movies' : set()
      },
    'dev' : {
        'mask' : np.zeros((num_users, num_movies), dtype=np.float32),
        'users' : set(),
        'movies' : set()
      },
    }

  user_idxs = {}
  movie_idxs = {}

  def get_user_idx(user_id):
    """ convert user_id into unique number in range (0~len(num_users))
    Args : 
      user_id : user's id in csv file
    returns :
      users_idxs[user_id] : converted user_id"""
    if user_id not in user_idxs:
      user_idxs[user_id] = len(user_idxs)
    return user_idxs[user_id]

  def get_movie_idx(movie_id):
    """ convert movie_id into unique number in range(0 ~ len(num_movies))
    Args :
      movie_id : movie id in csv file
    returns :
      movie_idxs[movie_id] : converted movie_id"""
    if movie_id not in movie_idxs:
      movie_idxs[movie_id] = len(movie_idxs)
    return movie_idxs[movie_id]

  for indices, k in [(train_indices, 'train'), (test_indices, 'test'), (dev_indices, 'dev')]:
    for row in rating_data.iloc[indices].itertuples():
      user_idx = get_user_idx(row.userId)
      movie_idx = get_movie_idx(row.movieId)
      dataset['rating'][user_idx, movie_idx] = row.rating
      dataset[k]['mask'][user_idx, movie_idx] = 1
      dataset[k]['users'].add(user_idx)
      dataset[k]['movies'].add(movie_idx)
    print("processed {} data".format(k))
    print("{} data shape {}".format(k, dataset[k]['mask'].shape))
    with open('{}_data.pkl'.format(k), 'wb') as data_path:
        pickle.dump(dataset[k], data_path, protocol=2)
    print("dumped {} data".format(k))
  print("write data into pkl")

if __name__=='__main__':
  data = load_data('dataset/ratings.csv')
  print(data)
