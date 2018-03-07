import os
import numpy as np
import tensorflow as tf

import net
import tools
from dataset.inputs import train_batch, val_batch
from config import ParseConfig, configure_session


def run(args_input, args_net, args_log):
    
    # Input
    train_file = ['data/train.tfrecords']
    val_file = ['data/val.tfrecords']   
    train_image_batch, train_label_batch = train_batch(train_file, batch_size=64)
    val_image_batch, val_label_batch = val_batch(val_file, batch_size=128)

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y_ = tf.placeholder(tf.int16, shape=[None, 2])    

    # Model Creation
    network = net.catalogue[args_net['net']](args_net['num_classes'],
                                                args_net['weight_decay'],
                                                args_net['batch_norm_decay'])
    
    logits = network.build(x, is_training=True)
    loss = tools.softmax_cross_entropy_with_logits(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimize(loss, args_net['learning_rate'], my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()   

    init = tf.global_variables_initializer()
    sess = tf.Session(config=configure_session())
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tra_summary_writer = tf.summary.FileWriter(args_log['train_log_dir'], sess.graph)
    val_summary_writer = tf.summary.FileWriter(args_log['val_log_dir'], sess.graph)

    try:
        MAX_STEP = args_net['max_step']
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            if step == 8000:
                train_op = tools.optimize(loss, 0.001, my_global_step)
            elif step == 26000:
                train_op = tools.optimize(loss, 0.0001, my_global_step)
                   
            train_images, train_labels = sess.run([train_image_batch, train_label_batch])
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={x:train_images, y_:train_labels})           
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op, feed_dict={x:train_images, y_:train_labels})
                tra_summary_writer.add_summary(summary_str, step)
                
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x:val_images, y_:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={x:train_images, y_:train_labels})
                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(args_log['train_log_dir'], 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()


if __name__ == "__main__":
    json_file = "config.json"
    cfg = ParseConfig(json_file)
    run(cfg.input, cfg.optimzer, cfg.log)
