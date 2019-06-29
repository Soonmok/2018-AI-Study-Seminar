import fire
import numpy as np 
import pandas as pd
import _pickle as pickle
from sklearn.model_selection import train_test_split
import scipy.sparse
from scipy.sparse import coo_matrix
import tensorflow as tf

def load_data(data_dir):
  """ load csv file and construct dataset,write dataset into sparse matrix file(.npz files)
      Args :
        file_path : csv datafile path """

  rating_data = pd.read_csv(data_dir + '/ratings.csv', sep=',', 
         names=['userId', 'movieId', 'rating', 'timestamp'])
  rating_data.rating = rating_data.rating.apply(pd.to_numeric, errors='coerce')

  indices = range(len(rating_data))
  train_indices, test_indices = train_test_split(indices, shuffle=True)
  print("loading total data{} train {} test {}".format(
        len(indices), len(train_indices), len(test_indices)))

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
  for indices, k in [(train_indices, 'train'), (test_indices, 'test')]: 
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
    with open(data_dir + '/{}_data.npz'.format(k), 'wb') as data_path:
        dataset = coo_matrix(
          (sparse_val, (sparse_row, sparse_col)), 
          shape=(num_users, num_movies))
        sparse_mat = scipy.sparse.csc_matrix(dataset)
        scipy.sparse.save_npz(data_path, sparse_mat)
    print("dumped {} data".format(k))
  print("dump rating data")
  with open(data_dir + '/rating_data.npz', 'wb') as data_path:
      total_dataset = coo_matrix(
          (total_sparse_val, (total_sparse_row, total_sparse_col)), 
          shape=(num_users, num_movies))
      sparse_mat = scipy.sparse.csc_matrix(total_dataset)
      scipy.sparse.save_npz(data_path, sparse_mat)

def get_sparse_matrix(numpy_array):
    """ convert numpy array into sparse matrix
    Args : 
        numpy_array: np array to convert
    Returns :
        indices_2d: indices where sparse matrix has nonzero value
        values :nonzero value list"""

    indices_2d = []
    values = ()
    for idx_row, row in enumerate(numpy_array):
        for idx_col, element in enumerate(row):
            if element != 0:
                indices_2d.append([idx_row, idx_col])
                values = values + (element,)
    return indices_2d, values

def sparse_train_generator(sparse_rating, sparse_train):
    """ generator to convert sparse matrix into np array for training
    Args : 
        sparse_rating: rating sparse matrix to convert (scipy csc matrix type)
        sparse_train: train sparse matrix to convert (scipy csc matrix type)
    Returns :
        rating_row : np array row of rating values
        ex) [4, 2, 0, 0, 1]
        train_row : np array row of train mask 
        ex) [1, 0, 0, 0, 1]""" 

    for rating_row, train_row in zip(sparse_rating, sparse_train):
        rating_row = np.asarray(rating_row.todense())[0]
        train_row = np.asarray(train_row.todense())[0]
        yield (rating_row, train_row) 

def sparse_test_generator(sparse_rating, sparse_train, sparse_test):
    """ generator to convert sparse matrix into np array for testing
    Args :
        sparse_rating: rating sparse matrix to convert (scipy csc matrix type)
        sparse_train: train sparse matrix to convert (scipy csc matrix type)
        sparse_test: test sparse matrix to convert (scipy csc matrix type)
    Returns :
        rating_row : np array row of rating values
        ex) [4, 2, 0, 0, 1]
        train_row : np array row of train mask 
        ex) [1, 0, 0, 0, 1]
        test_row : np.array row of test mask
        ex) [0, 1, 0, 0, 1]""" 

    for rating_row, train_row, test_row in zip(sparse_rating, sparse_train, sparse_test):
        rating_row = np.asarray(rating_row.todense())[0]
        train_row = np.asarray(train_row.todense())[0]
        test_row = np.asarray(test_row.todense())[0]
        yield (rating_row, train_row, test_row)

def get_train_dataset(sparse_rating, sparse_matrix, batch_size):
    """ get tensorflow dataset for training"""
    dataset = tf.data.Dataset.from_generator(
        lambda: sparse_train_generator( sparse_rating, sparse_matrix), 
        (tf.float32, tf.float32))
    return dataset.batch(batch_size).repeat()

def get_test_dataset(sparse_rating, sparse_train, sparse_matrix, batch_size):
    """ get tensorflow dataset for testing"""
    dataset = tf.data.Dataset.from_generator(
        lambda: sparse_test_generator(
            sparse_rating, sparse_train, sparse_matrix), 
        (tf.float32, tf.float32, tf.float32))
    return dataset.batch(batch_size).repeat()


if __name__=='__main__':
  fire.Fire(load_data('dataset/ratings.csv'))

