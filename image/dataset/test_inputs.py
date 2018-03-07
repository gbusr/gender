import cv2
import numpy as np
import tensorflow as tf

from inputs import train_batch, val_batch


train_file = ['data/train.tfrecords']
val_file = ['data/val.tfrecords']

train_image_batch, train_label_batch = train_batch(train_file, batch_size=64)
val_image_batch, val_label_batch = val_batch(val_file, batch_size=128)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    train_images, train_labels, val_images, val_labels = sess.run(
        [train_image_batch, train_label_batch, val_image_batch, val_label_batch])
    coord.request_stop()
    coord.join(threads)

print(np.shape(train_images))
print(np.shape(train_labels))
print(np.shape(val_images))
print(np.shape(val_labels))

for i in range(5):
    image_bgr = train_images[i]
    cv2.putText(image_bgr, "gender: {}".format(train_labels[i]), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.imshow("train_image", image_bgr)
    cv2.waitKey()

for i in range(5):
    image_bgr = val_images[i]
    cv2.putText(image_bgr, "gender: {}".format(val_labels[i]), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.imshow("val_image", image_bgr)
    cv2.waitKey()

# image_bgr = cv2.resize(image[0],(480, 480))
# image_bgr = image[0]
# print(image_bgr)
# cv2.putText(image_bgr, "gender: {}".format(label[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
# cv2.imshow("image", image_bgr)
# cv2.waitKey()
