import tensorflow as tf
import numpy as np
import sys
sys.path.append('../funcs/')
import nn

class VGG(object):        
    feature_dim  = 2
    weight_decay_factor = 1e-6
    learning_rate = 5e-5
    def __init__(self, weight_decay, learning_rate, feature_dim = feature_dim, label_dim = 2, maxout = False):
        self.x_train = tf.placeholder(tf.float32, [None, 224, 224, feature_dim])
        self.y_train = tf.placeholder(tf.uint8, [None, label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, 224, 224, feature_dim])
        self.y_test = tf.placeholder(tf.uint8, [None, label_dim])
        
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        _, self.output = self.vgg(self.x_train, 0.5)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, 
                                                                           labels = self.y_train))
        
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        _, self.train_pred = self.vgg(self.x_train, keep_prob = 1.0, reuse = True)
        self.train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.train_pred, 1), 
                                                              tf.argmax(self.y_train, 1)), tf.float32))
        _, self.val_pred = self.vgg(self.x_test, keep_prob = 1.0, reuse = True)
        self.val_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.val_pred, 1), 
                                                        tf.argmax(self.y_test, 1)), tf.float32))
           
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.train_summary = tf.summary.scalar('training_accuracy', self.train_accuracy)
    
    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    # Training Accuracy
    def train_validate(self, sess, x_train, y_train):
        train_accuracy, train_summary = sess.run((self.train_accuracy, self.train_summary), 
                                     feed_dict={self.x_train: x_train, self.y_train: y_train})
        return train_accuracy, train_summary
    
    # Validation Accuracy
    def validate(self, sess, x_test, y_test):
        val_accuracy= sess.run((self.val_accuracy), feed_dict={self.x_test: x_test, self.y_test: y_test})
        return val_accuracy
    
    def predict(self, sess, x):
        prediction = sess.run((self.val_pred), feed_dict={self.x_test: x})
        return np.argmax(prediction, axis = 1)
   
    def probabilities(self, sess, x):
        probability = sess.run((tf.nn.softmax(self.val_pred)), feed_dict={self.x_test: x})
        return probability
 
    def vgg(self, input, keep_prob = 0.5, reuse = None):
        with tf.variable_scope('vgg', reuse=reuse):
            pool_ = lambda x: nn.max_pool(x, 2, 2)
            max_out_ = lambda x: nn.max_out(x, 16)
            conv_ = lambda x, output_depth, name, trainable = True: nn.conv(x, 3, output_depth, 1, self.weight_decay, name=name, trainable = trainable)
            fc_ = lambda x, features, name, relu = True: nn.fc(x, features, self.weight_decay, name, relu = relu)
            VGG_MEAN = [103.939, 116.779, 123.68]
            VGG_MEAN = [24.0, 24.0, 24.0]
            # Convert RGB to BGR and subtract mean
            # red, green, blue = tf.split(input, 3, axis=3)
            input = tf.concat([
                input - VGG_MEAN[0],
                input - VGG_MEAN[1],
                input - VGG_MEAN[2],
            ], axis=3)

            conv_1_1 = conv_(input, 64, 'conv1_1', trainable = True)
            conv_1_2 = conv_(conv_1_1, 64, 'conv1_2', trainable = True)

            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, 128, 'conv2_1', trainable = True)
            conv_2_2 = conv_(conv_2_1, 128, 'conv2_2', trainable = True)

            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, 256, 'conv3_1')
            conv_3_2 = conv_(conv_3_1, 256, 'conv3_2')
            conv_3_3 = conv_(conv_3_2, 256, 'conv3_3')

            pool_3 = pool_(conv_3_3)

            conv_4_1 = conv_(pool_3, 512, 'conv4_1')
            conv_4_2 = conv_(conv_4_1, 512, 'conv4_2')
            conv_4_3 = conv_(conv_4_2, 512, 'conv4_3')

            pool_4 = pool_(conv_4_3)

            conv_5_1 = conv_(pool_4, 512, 'conv5_1')
            conv_5_2 = conv_(conv_5_1, 512, 'conv5_2')
            conv_5_3 = conv_(conv_5_2, 512, 'conv5_3')
            
            pool_5 = pool_(conv_5_3)
            if self.maxout:
                max_5 = max_out_(pool_5)
                flattened = tf.contrib.layers.flatten(max_5)
            else:
                flattened = tf.contrib.layers.flatten(pool_5)
            
            fc_6 = nn.dropout(fc_(flattened, 4096, 'fc6'), keep_prob)
            fc_7 = nn.dropout(fc_(fc_6, 4096, 'fc7'), keep_prob)
            fc_8 = fc_(fc_7, self.label_dim, 'fc8', relu=False)
            return fc_7, fc_8

    def siamese(self, input1, input2, keep_prob = 0.5, weight_decay = \
                weight_decay_factor, reuse = None):
        fc_ = lambda x, features, name, relu = True: nn.fc(x, features, weight_decay, name, relu = relu)
        feat1, _ = self.vgg(input1, keep_prob, True)
        feat2, _ = self.vgg(input2, keep_prob, True)
        with tf.variable_scope('network', reuse = reuse):
            fc_combined = tf.concat((feat1,feat2),1)
            fc_8 = nn.dropout(fc_(fc_combined, 4096, 'fc8'), keep_prob)
            fc_9 = fc_(fc_8, 2, 'fc9', relu = False)
            return fc_9
        
    def init_weights(self, sess, vgg_file):
        weights_dict = np.load(vgg_file, encoding='bytes').item()
        weights_dict = { key.decode('ascii') : value for key, value in weights_dict.items() }
        with tf.variable_scope('vgg', reuse=True):
            for layer in ['conv1_1', 'conv1_2',
                          'conv2_1', 'conv2_2']:
                with tf.variable_scope(layer):
                    W_value, b_value = weights_dict[layer]
                    W = tf.get_variable('W', trainable = False)
                    b = tf.get_variable('b', trainable = False)
                    sess.run(W.assign(W_value))
                    sess.run(b.assign(b_value))
        with tf.variable_scope('vgg', reuse=True):
            for layer in ['conv3_1', 'conv3_2', 'conv3_3',
                          'conv4_1', 'conv4_2', 'conv4_3',
                          'conv5_1', 'conv5_2', 'conv5_3']:
                with tf.variable_scope(layer):
                    W_value, b_value = weights_dict[layer]
                    W = tf.get_variable('W')
                    b = tf.get_variable('b')
                    sess.run(W.assign(W_value))
                    sess.run(b.assign(b_value))


