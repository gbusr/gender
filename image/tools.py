import numpy as np
import tensorflow as tf


def softmax_cross_entropy_with_logits(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss
    

def sparse_softmax_cross_entropy(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                logits=logits)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss


def accuracy(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
  """
  with tf.name_scope('accuracy') as scope:
      correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct) * 100.0
      tf.summary.scalar(scope+'/accuracy', accuracy)
  return accuracy


def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

