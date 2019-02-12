import numpy as np 
import pandas as pd
import _pickle as pickle
from sklearn.model_selection import train_test_split
import scipy.sparse

def batch_iter(dataset, batch_size):
    """ make batch dataset for learning 
        Args :
            dataset : dataset to divide into batch segment
            batch_size : batch_size
        returns :
            indices_2d : non zero sparse matrix indices_2d
            values : non zero value of sparse matrix (match to 2d_indices)
            shape : shape of batch sparse matrix"""

    for idx in range(int(dataset.shape[0] / batch_size)):
        start_idx = idx * batch_size 
        end_idx = min((idx + 1) * batch_size, len(dataset)-1)
        batch = dataset[start_idx:end_idx]
        input_mask = (batch != 0).astype(np.float32)
        indices_2d = []
        values = ()
        shape = (input_mask.shape[0], input_mask.shape[1])
        for idx_row, row in enumerate(batch):
            row = np.asarray(row)
            row = np.squeeze(row, axis=0)
            for idx_col, element in enumerate(row):
                if element != 0:
                    indices_2d.append([idx_row, idx_col])
                    values = values + (element,)
        yield zip(indices_2d, values), shape
            

def load_data(file_path):
  """ load csv file and create dataset for deep learning
      Args :
        file_path : csv datafile path
      returns:
        dataset : shape(user_num, movie_num) matrix devided into train, test, dev set"""

  rating_data = pd.read_csv(file_path, sep='::', 
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

  total_dataset = np.zeros((num_users, num_movies))
  for indices, k in [(train_indices, 'train'), (test_indices, 'test'), (dev_indices, 'dev')]:
    dataset = np.zeros((num_users, num_movies), dtype=np.float32)
    for row in rating_data.iloc[indices].itertuples():
      user_idx = get_user_idx(row.userId)
      movie_idx = get_movie_idx(row.movieId)
      dataset[user_idx, movie_idx] = 1 
      total_dataset[user_idx, movie_idx] = row.rating
    print("processed {} data".format(k))
    with open('{}_data.npz'.format(k), 'wb') as data_path:
        sparse_mat = scipy.sparse.csc_matrix(dataset)
        scipy.sparse.save_npz(data_path, sparse_mat)
    print("dumped {} data".format(k))
  print("dump rating data")
  with open('rating_data.npz', 'wb') as data_path:
      sparse_mat = scipy.sparse.csc_matrix(total_dataset)
      scipy.sparse.save_npz(data_path, sparse_mat)

def sparse_generator(rating_data, indices, num_movies):
  for row in rating_data.iloc[train_indices].itertuples():
      user_idx = get_user_idx(row.userId)
      movie_idx = get_movie_idx(row.movieId)
      indices_2d = (user_idx, movie_idx)
      value = (row.rating, )
      shape = (1, num_movies)
      yield (indices_2d, value, shape) 

def get_dataset():
    dataset = tf.data.Dataset.from_generator(sparse_generator, (tf.int64, tf.float32, tf.int64))
    dataset = dataset.map(lambda i, v, s: tf.SparseTensor(i, v, s))
    return dataset


if __name__=='__main__':
  data = load_data('dataset/ratings.csv')
  print(data)
