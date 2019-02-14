from model import AutoEncoder
from data_load import *
import tensorflow as tf
import argparse
from data_load import load_data, batch_iter, get_dataset, get_sparse_matrix
import _pickle as pickle
import os
import scipy.sparse
from tqdm import tqdm

if __name__=="__main__":
    # setting hyper parameters
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='./ml-1m/ratings.dat')
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--hidden_size', type=int, default=500)
    args.add_argument('--penalty', type=float, default=0.001)
    args.add_argument('--learning_rate', type=float, default=0.001)
    args.add_argument('--epoch', type=int, default=50)
    config = args.parse_args()

    # load data
    if not os.path.exists("./rating_data.npz"):
        load_data('./ml-1m/ratings.dat')
    sparse_total_dataset = scipy.sparse.load_npz("./rating_data.npz")
    sparse_train_dataset = scipy.sparse.load_npz("./train_data.npz")
    sparse_test_dataset = scipy.sparse.load_npz("./test_data.npz")
    num_users = sparse_total_dataset.get_shape()[0]
    num_movies = sparse_total_dataset.get_shape()[1]
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

    # make input match into train/test dataset
    X_dense = tf.multiply(tf.sparse.to_dense(input_mask),
                          tf.sparse.to_dense(ratings))
    model = AutoEncoder(X_dense, config.hidden_size)

    reconstrunction_cost = tf.losses.mean_squared_error(
       tf.multiply(model.X_dense_reconstructed, tf.sparse.to_dense(output_mask)), 
       tf.sparse.to_dense(ratings))
    reg_cost = tf.reduce_sum(model.w_decoder_1 ** 2) + tf.reduce_sum(model.w_decoder_2 ** 2) + tf.reduce_sum(model.w_encoder_1 ** 2) + tf.reduce_sum(model.w_encoder_2 ** 2)
    total_cost = reconstrunction_cost + 0.5 * reg_cost * config.penalty 
    train_op = tf.train.AdamOptimizer(
        config.learning_rate).minimize(total_cost, global_step=global_step)

    """ -------------- model part end---------------------"""

    # start session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    print("initialize model")
    init = tf.global_variables_initializer()
    sess.run(init)

    # get tf.dataset_from_generator
    train_dataset = get_dataset(
        sparse_total_dataset, sparse_train_dataset, batch_size = config.batch_size)
    test_dataset = get_test_dataset(
        sparse_total_dataset, sparse_train_dataset,
        sparse_test_dataset, batch_size = config.batch_size)

    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()

    train_batch_tf = train_iterator.get_next()
    test_batch_tf = test_iterator.get_next()

    def train_step(train_batch_tf):
        """ train batches 
        Args : 
            train_batch_tf: batch of tf train data (rating_row, train_row))
            ex) ([2, 0, 0, 1 ...], [1, 0, 0, 1 ...])
        Returns :
            cost : loss value
            step : train step number"""
        
        output = sess.run(train_batch_tf)
        rating_val, train_val = output[0], output[1]
        rating_indices_2d, rating_values = get_sparse_matrix(rating_val)
        train_indices_2d, one_values = get_sparse_matrix(train_val)

        feed_dict = {input_mask : tf.SparseTensorValue(
            train_indices_2d, one_values, rating_val.shape),
                     output_mask : tf.SparseTensorValue(
            train_indices_2d, one_values, rating_val.shape),
                     ratings : tf.SparseTensorValue(
            rating_indices_2d, rating_values, rating_val.shape)}

        _, cost, step = sess.run([train_op, total_cost, global_step], feed_dict)
        return cost, step

    def _test_step(test_batch_tf):
        """helper function of test step
        Args : 
            test_batch_tf: batch of tf test data (rating_row, test_row)
            ex) ([2, 0, 0, 1 ...], [1, 0, 0, 0 ...])
        Returns :
            cost : loss value of test batch data"""

        output = sess.run(test_batch_tf) 
        rating_val, train_val, test_val = output[0], output[1], output[2]

        rating_indices_2d, rating_values = get_sparse_matrix(rating_val) 
        train_indices_2d, train_values = get_sparse_matrix(train_val)
        test_indices_2d, test_values = get_sparse_matrix(test_val) 

        feed_dict = {input_mask : tf.SparseTensorValue(
            train_indices_2d, train_values, test_val.shape),
                    output_mask : tf.SparseTensorValue(
            test_indices_2d, test_values, test_val.shape),
                    ratings : tf.SparseTensorValue(
            rating_indices_2d, rating_values, test_val.shape)}

        cost = sess.run(total_cost, feed_dict)
        return cost

    def test_step(test_batch_tf):
        """test step
        Args : 
            test_batch_tf : batch of tf test data (rating_row, test_row)
            ex) ([2, 0, 0, 1 ...], [1, 0, 0, 0 ...])
        Returns :
            cost : mean loss of total test data"""
        total_test_cost = 0
        counter = 0
        print("Testing model ...")
        pbar = tqdm(total=num_users/config.batch_size)
        while counter < int(num_users/config.batch_size-1):
            try:
                counter += 1
                total_test_cost += _test_step(test_batch_tf) 
                pbar.update(1)
            except tf.errors.InvalidArgumentError:
                break
        return total_test_cost / counter

    # training part
    epoch = 1
    while True:
        try:
            train_cost, step_num = train_step(train_batch_tf)
            print("step : {}, cost : {}".format(step_num, train_cost))
            if step_num % (num_users/config.batch_size-1) == 0:
                test_cost = test_step(test_batch_tf)
                print("=====================================")
                print("Test cost == > {}".format(test_cost))
        except tf.errors.InvalidArgumentError:
            test_cost = test_step(test_batch_tf)
            print("{} epoch Test cost == > {}".format(epoch, test_cost))
            epoch += 1
            if epoch < config.batch_size:
                pass
            else:
                print("Finish")
                break








