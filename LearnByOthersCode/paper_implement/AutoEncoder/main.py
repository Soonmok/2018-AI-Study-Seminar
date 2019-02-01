from model import AutoEncoder

if __name__=="__main__":
    X = tf.placeholder(tf.float32, shape(10000, 10000))
    model = AutoEncoder(X)