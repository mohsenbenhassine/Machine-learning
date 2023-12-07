import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim=10):
        self.input_dim = input_dim #A
        self.seq_size = seq_size #A
        self.hidden_dim = hidden_dim #A
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out') #B
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out') #B
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim]) #B
        self.y = tf.placeholder(tf.float32, [None, seq_size]) #B
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y)) #C
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost) #C
        self.saver = tf.train.Saver() #D
    def model(self):
            cell = rnn.BasicLSTMCell(self.hidden_dim) #A
            outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32) #B
            num_examples = tf.shape(self.x)[0]
            W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
            out = tf.matmul(outputs, W_repeated) + self.b_out
            out = tf.squeeze(out)
            return out
    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000): #A
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x,self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)
            save_path = self.saver.save(sess, 'model.ckpt')
            print('Model saved to {}'.format(save_path))
    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            print(output)

if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
    train_x = [[[1], [2], [5], [6]],
    [[5], [7], [7], [8]],
    [[3], [4], [5], [7]]]
    train_y = [[1, 3, 7, 11],
    [5, 12, 14, 15],
    [3, 7, 9, 12]]
    predictor.train(train_x, train_y)
    test_x = [[[1], [2], [3], [4]], #A
    [[4], [5], [6], [7]]] #B
    predictor.test(test_x)
    