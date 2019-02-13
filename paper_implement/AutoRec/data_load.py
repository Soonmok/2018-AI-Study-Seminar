import numpy as np 
import pandas as pd
import _pickle as pickle
from sklearn.model_selection import train_test_split
import scipy.sparse
from scipy.sparse import coo_matrix
import tensorflow as tf

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
  """ load csv file and construct dataset,write dataset into sparse matrix file(.npz files)
      Args :
        file_path : csv datafile path """

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

  # change data into (num_users, num_movies) shape sparse matrix and dump it 
  total_sparse_row, total_sparse_col, total_sparse_val = [], [], []
  for indices, k in [(train_indices, 'train'), (test_indices, 'test'), (dev_indices, 'dev')]:
    sparse_row, sparse_col, sparse_val = [], [], []
    for row in rating_data.iloc[indices].itertuples():
      user_idx = get_user_idx(row.userId)
      movie_idx = get_movie_idx(row.movieId)
      sparse_row.append(user_idx)
      sparse_col.append(movie_idx)
      sparse_val.append(1)
      total_sparse_row.append(user_idx)
      total_sparse_col.append(movie_idx)
      total_sparse_val.append(row.rating)
    print("processed {} data".format(k))
    with open('{}_data.npz'.format(k), 'wb') as data_path:
        dataset = coo_matrix(
          (sparse_val, (sparse_row, sparse_col)), 
          shape=(num_users, num_movies))
        sparse_mat = scipy.sparse.csc_matrix(dataset)
        scipy.sparse.save_npz(data_path, sparse_mat)
    print("dumped {} data".format(k))
  print("dump rating data")
  with open('rating_data.npz', 'wb') as data_path:
      total_dataset = coo_matrix(
          (total_sparse_val, (total_sparse_row, total_sparse_col)), 
          shape=(num_users, num_movies))
      sparse_mat = scipy.sparse.csc_matrix(total_dataset)
      scipy.sparse.save_npz(data_path, sparse_mat)

def get_sparse_matrix(numpy_array):
    indices_2d = []
    values = ()
    for idx_row, row in enumerate(numpy_array):
        for idx_col, element in enumerate(row):
            if element != 0:
                indices_2d.append([idx_row, idx_col])
                values = values + (element,)
    return indices_2d, values

def sparse_generator(sparse_matrix):
    for element in sparse_matrix:
        row = np.asarray(element.todense())[0]
        yield row 

def get_dataset(sparse_matrix, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: sparse_generator(sparse_matrix), tf.float32)
    return dataset.batch(batch_size).repeat()


if __name__=='__main__':
  data = load_data('dataset/ratings.csv')
  print(data)
