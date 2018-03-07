import os
import sys
import cv2
import dlib
import time
import argparse
import numpy as np

import tensorflow as tf
from imutils import face_utils
from imdb import IMDB, split_imdb_data


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, age, gender):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': _bytes_feature(filename),
        'image/encoded': _bytes_feature(image_buffer),
        'image/class/age': _int64_feature(age),
        'image/class/gender': _int64_feature(gender)
    }))
    return example


def _align_face_with_dlib(filename):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    shape_predictor = 'dataset/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    face_align = face_utils.FaceAligner(predictor, desiredFaceWidth=224)

    try:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)

        if len(rects) == 1:
            image_raw = face_align.align(image, gray, rects[0])
            image_raw = image_raw.tostring()
            return image_raw
        else:
            image_raw = None
    except IOError:  # some files seem not exist in face_data dir
        print("filename")

    return image_raw


def convert_to_tfrecord(dataset, data_path, output_path, name):
    filenames = dataset.filename
    genders = dataset.gender
    ages = dataset.age

    num_samples = len(filenames)
    num_examples = np.arange(num_samples)
    np.random.shuffle(num_examples)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tfrecord_file = os.path.join(output_path, name + '.tfrecords')
    num = 1
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # print(num_examples)
    for i in num_examples:
        filename = os.path.join(data_path, filenames[i])
        image_raw = _align_face_with_dlib(filename)
        if image_raw is not None:
            example = _convert_to_example(filename.encode('utf8'),
                                          image_raw,
                                          int(ages[i]),
                                          int(genders[i]))
            writer.write(example.SerializeToString())
            num = num + 1
            sys.stdout.write('\r>> Converting image %d/%d' % (num, num_samples))
            sys.stdout.flush()
        else:
            pass
    writer.close()
    # print('finish ', tfrecord_file)
    # print("%d valid faces, %d total faces" %(num, num_examples))


def main(data_path, output_path):
    start_time = time.time()
    db = IMDB()
    data = db.get_data()
    train_sets, val_sets = split_imdb_data(data, 0.3)

    # convert_to_tfrecord(train_sets, data_path, output_path, 'train')
    convert_to_tfrecord(val_sets, data_path, output_path, 'val')
    duration = time.time() - start_time
    print("Running %.3f sec All done!" % duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/cv/Disk/data/face/imdb_crop")
    parser.add_argument("--output_path", type=str, default="data/")
    args = parser.parse_args()

    main(args.data_path, args.output_path)
