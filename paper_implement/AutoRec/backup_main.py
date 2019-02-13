from model import AutoEncoder
from data_load import *
import tensorflow as tf
import argparse
from data_load import load_data, batch_iter
import _pickle as pickle
import os
import scipy.sparse

if __name__=="__main__":
    # setting hyper parameters
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--hidden_size', type=int, default=500)
    args.add_argument('--penalty', type=float, default=0.001)
    args.add_argument('--learning_rate', type=float, default=0.001)
    args.add_argument('--epoch', type=int, default=50)
    config = args.parse_args()

    # load data
    if not os.path.exists("./rating_data.npz"):
        load_data('./dataset/ratings.csv')
    sparse_ratings = scipy.sparse.load_npz("./rating_data.npz")
    sparse_train_dataset = scipy.sparse.load_npz("./train_data.npz")
    sparse_test_dataset = scipy.sparse.load_npz("./test_data.npz")
    sparse_dev_dataset = scipy.sparse.load_npz("./dev_data.npz")

    # change sparse matrix into dense matrix
    ratings_data = sparse_ratings.todense()
    train_dataset = sparse_train_dataset.todense()
    test_dataset = sparse_test_dataset.todense()
    dev_dataset = sparse_dev_dataset.todense()

    num_movies = ratings_data.shape[1]
    print("rating shape {}".format(ratings_data.shape))
    print("model construction")

    """--------------- model part start -------------------"""
    # construct model
    input_mask = tf.sparse.placeholder(
        tf.float32, shape=np.array([config.batch_size, num_movies], dtype=np.int64))
    output_mask = tf.sparse.placeholder(
        tf.float32, shape=np.array([config.batch_size, num_movies], dtype=np.int64))
    ratings = tf.sparse.placeholder(
        tf.float32, shape=np.array([config.batch_size, num_movies], dtype=np.int64))
    global_step = tf.Variable(0, name="global_step")

    # make input match into train/test/dev dataset
    X_dense = tf.multiply(tf.sparse.to_dense(input_mask),
                          tf.sparse.to_dense(ratings))
    model = AutoEncoder(X_dense, config.hidden_size)

    reconstrunction_cost = tf.losses.mean_squared_error(
       tf.multiply(model.X_dense_reconstructed, tf.sparse.to_dense(output_mask)), 
       tf.sparse.to_dense(ratings))
    reg_cost = tf.reduce_sum(model.w_encoder ** 2) + tf.reduce_sum(model.w_decoder ** 2)
    total_cost = reconstrunction_cost + 0.5 * reg_cost * config.penalty 
    train_op = tf.train.AdamOptimizer(
        config.learning_rate).minimize(total_cost, global_step=global_step)

    """ -------------- model part end---------------------"""
    
    # start session
    sess = tf.Session()
    print("initialize model")
    init = tf.global_variables_initializer()
    sess.run(init)

    def train_step(train_indices, train_values, train_shape):
        """ train batches 
        Args : 
            train_indices : indices of values in sparse matrix batch
            train_values : values of sparse matrix batch
            train_shape : shape of batch sparse matrix shape
           
            ex) [[0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]]  
            --> train_indices = [(0,2), (1,0), (2,1)]
            --> train_values = [1, 1, 1]
            --> train_shape = (3, 3)
        Returns :
            cost : loss value
            step : train step number"""

        one_values = np.array([1] * len(train_values))
        feed_dict = {input_mask : tf.SparseTensorValue(
            train_indices, one_values, train_shape),
                     output_mask : tf.SparseTensorValue(
            train_indices, one_values, train_shape),
                     ratings : tf.SparseTensorValue(
            train_indices, train_values, train_shape)}
        _, cost, step = sess.run([train_op, total_cost, global_step], feed_dict)
        return cost, step

    def dev_step(dev_indices, dev_values, dev_shape):
        """ inference dev batches (not train !!)
        Args :
            dev_indices : indices of values in sparse matrix batch
            dev_values : values of sparse matrix batch
            dev_shpae : shape of batch sparse matrix shape
        Returns :
            cost : difference between predicted matrix and real matrix"""

        one_values = np.array([1] * len(dev_values))
        feed_dict = {input_mask : tf.SparseTensorValue(
            dev_indices, one_values, dev_shape),
                    output_mask : tf.SparseTensorValue(
            dev_indices, one_values, dev_shape),
                    ratings : tf.SparseTensorValue(
            dev_indices, dev_values, dev_shape)}
        cost = sess.run(total_cost, feed_dict)
        return cost

    # training part
    for epoch in range(config.epoch):
        batches = batch_iter(ratings_data, config.batch_size)
        for batch, shape in batches:
            indices_2d, values = zip(*batch)
            train_cost, step_num = train_step(indices_2d, values, shape)
            print("epoch : {}, step : {}, cost : {}".format(epoch, step_num, train_cost))
            if step_num % 20 == 0:
                dev_batches = batch_iter(dev_dataset, config.batch_size)
                dev_cost, counter = 0, 0
                for dev_batch, dev_shape in dev_batches:
                    dev_indices_2d, dev_values = zip(*dev_batch)
                    dev_cost += dev_step(
                        dev_indices_2d, dev_values, dev_shape)
                    counter += 1
                print("dev MSE : {})".format(dev_cost/counter))
        test_batches = batch_iter(test_dataset, config.batch_size)
        test_cost, counter = 0, 0
        for test_batch, test_shape in test_batches:
            test_indices_2d, test_values = zip(*test_batch)
            test_cost += dev_step(
                    test_indices_2d, test_values, test_shape)
            counter += 1
        print("=============================================")
        print("epoch : {}, test MSE score == {}".format(
            epoch, test_cost/counter))








