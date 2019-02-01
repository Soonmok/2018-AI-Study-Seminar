from model import AutoEncoder
from data_load import *

if __name__=="__main__":
    X = tf.placeholder(tf.float32, shape(10000, 10000))
    model = AutoEncoder(X)
    real_X = X
    
    reconstrunction_cost = tf.losses.mean_squared_error(real_X, model.X_reconstructed)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(reconstrunction_cost)
    
    train_CF_data = 
    
    with tf.Session as sess:
        sess.run(tf.global_variables_initializer())
        for batch in batch_iter(train_CF_data, batch_size, total_epoch, shuffle=True):
            _, cost = sess.run([optimizer, reconstrunction_cost], feed_dict={X: batch})
            print("cost : {:g}".format(cost))



    