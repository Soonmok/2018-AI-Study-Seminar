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
        load_data('./ml-1m/ratings.dat')
    sparse_ratings = scipy.sparse.load_npz("./rating_data.npz")
    ratings_data = sparse_ratings.todense()
    num_movies = ratings_data.shape[1]
    print("rating shape {}".format(ratings_data.shape))
    print("model construction")
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
    
    sess = tf.Session()
    print("initialize model")
    init = tf.global_variables_initializer()
    sess.run(init)

    def train_step(train_indices, train_values, train_shape):
        input_mask_sp = tf.SparseTensor(indices=train_indices,
                                        values=train_values,
                                        dense_shape=train_shape)
        output_mask_sp = tf.SparseTensor(indices=train_indices,
                                        values=train_values,
                                        dense_shape=train_shape)
        ratings_sp = tf.SparseTensor(indices=train_indices,
                                        values=train_values,
                                        dense_shape=train_shape)
        input_mask_val = input_mask_sp.eval(session=sess)
        output_mask_val = output_mask_sp.eval(session=sess)
        ratings_val = ratings_sp.eval(session=sess)
        one_values = np.array([1] * len(train_values))
        feed_dict = {input_mask : input_mask_val,
                     output_mask : output_mask_val,
                     ratings : ratings_val}
        _, cost, step = sess.run([train_op, total_cost, global_step], feed_dict)
        return cost, step

    for epoch in range(config.epoch):
        batches = batch_iter(ratings_data, config.batch_size)
        for batch, shape in batches:
            indices_2d, values = zip(*batch)
            train_cost, step_num = train_step(indices_2d, values, shape)
            print("epoch : {}, step : {}, cost : {}".format(epoch, step_num, train_cost))

