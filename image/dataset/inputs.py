import tensorflow as tf


def _parse_serialized_example(serialized_example):
    features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/age': tf.FixedLenFeature([], tf.int64),
        'image/class/gender': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string)
    }
    return tf.parse_single_example(serialized=serialized_example, features=features)


def _preprocess_example(raw_image, target_image_size, is_training=False):
    image = tf.decode_raw(raw_image, tf.uint8)
    image = tf.reshape(image, target_image_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    if is_training is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def _read_from_tfrecords(filename_queue, target_image_size, is_training):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    parsed_example = _parse_serialized_example(serialized_example)
    example = _preprocess_example(parsed_example['image/encoded'],
                                target_image_size, is_training)
    label = parsed_example['image/class/gender']
    return example, label


def _input(filenames, batch_size, target_image_size, 
            num_epochs, n_classes, shuffle, is_training):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=num_epochs,
                                                    shuffle=shuffle)
    example, label = _read_from_tfrecords(filename_queue, 
                                target_image_size, is_training)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    return example_batch, label_batch


def train_batch(filenames, batch_size, target_image_size=[224, 224, 3],
        num_epochs=None, n_classes=2, shuffle=True, is_training=True):
    return _input(filenames, batch_size, target_image_size, 
                    num_epochs, n_classes, shuffle, is_training)


def val_batch(filenames, batch_size, target_image_size=[224, 224, 3],
        num_epochs=None, n_classes=2, shuffle=False, is_training=False):
    return _input(filenames, batch_size, target_image_size, 
                    num_epochs, n_classes, shuffle, is_training)
