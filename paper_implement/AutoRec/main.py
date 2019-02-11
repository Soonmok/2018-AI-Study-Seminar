from model import AutoEncoder
from data_load import *
import tensorflow as tf
import argparse
from data_load import load_data, batch_iter
import _pickle as pickle
import os

if __name__=="__main__":
    # test
    # setting hyper parameters
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--hidden_size', type=int, default=500)
    args.add_argument('--penalty', type=float, default=0.001)
    args.add_argument('--learning_rate', type=float, default=0.001)
    args.add_argument('--epoch', type=int, default=50)
    config = args.parse_args()

    # load data
    if not os.path.exists("./rating_data.pkl"):
        load_data("./dataset/ratings.csv")
    with open("./train_data.pkl", 'rb') as data:
        dataset = pickle.load(data)

    num_movies = dataset['rating'].shape[1]
    print("model construction")
    # construct model
    """ -------------- model part start---------------------"""
    input_mask = tf.placeholder(
        dtype=tf.float32, shape=[None, num_movies], name="input_mask")
    output_mask = tf.placeholder(
        dtype=float32, shape=[None, num_movies], name="output_mask")
    ratings = tf.placeholder(
        dtype=tf.float32, shape=[None, num_movies], name="ratings")
    global_step = tf.Variable(0, name="global_step")

    # make input match into train/test/dev dataset
    input_X = ratings * input_mask
    model = AutoEncoder(input_X, config.hidden_size)

    reconstrunction_cost = tf.losses.mean_squared_error(
        ratings, model.rating_recontructed)
    reg_cost = tf.reduce_sum(model.w_encoder ** 2) + tf.reduce_sum(model.w_decoder ** 2)
    total_cost = reconstrunction_cost + 0.5 * reg_cost * config.penalty 
    train_op = tf.train.AdamOptimizer(
        config.learning_rate).minimize(total_cost, global_step=global_step)

    """ -------------- model part end---------------------"""
    
    sess = tf.Session()
    print("initialize model")
    init = tf.global_variables_initializer()
    sess.run(init)

    def train_step(rating_batch, mask_batch):
        feed_dict = {input_mask : mask_batch,
                     output_mask : mask_batch,
                     ratings : rating_batch}
        _, cost, train_step = sess.run([train_op, total_cost, global_step], feed_dict)
        return cost

    for epoch in range(config.epoch):
        batches = batch_iter(dataset, config.batch_size)
        for batch in batches:
            rating_batch, mask_batch = zip(*batch)
            _, train_cost, train_step = train_step(rating_batch, mask_batch)
            print("epoch : {}, step : {}, cost : {}".fotmat(epoch, train_step, train_cost))

