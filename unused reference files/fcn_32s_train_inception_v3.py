# % matplotlib inline

import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from matplotlib import pyplot as plt

# # YW: I don't think we have GPU
# # Use second GPU -- change if you want to use a first one
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# TO CHANGE
# Add a path to a custom fork of TF-Slim
# Get it from here:
# https://github.com/warmspringwinds/models/tree/fully_conv_vgg
sys.path.append("/home/dpakhom1/workspace/my_models/slim/")

# Add path to the cloned library
sys.path.append("/home/dpakhom1/tf_projects/segmentation/tf-image-segmentation/")

checkpoints_dir = '/home/dpakhom1/checkpoints'
log_folder = '/home/dpakhom1/tf_projects/segmentation/log_folder'


slim = tf.contrib.slim
inception_v3_checkpoint_path = os.path.join(checkpoints_dir, 'inception_v3.ckpt')

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
# TO CHANGE fcn_32s.py
from tf_image_segmentation.models.fcn_32s import FCN_32s

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

# YW: start without distort but don't how to do it
from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)

image_train_size = [384, 384]
number_of_classes = 21
tfrecord_filename = 'pascal_augmented_train.tfrecords'
pascal_voc_lut = pascal_segmentation_lut()
class_labels = pascal_voc_lut.keys()


filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=10)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Various data augmentation stages
image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

# image = distort_randomly_image_color(image)

resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)


resized_annotation = tf.squeeze(resized_annotation)

image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                             batch_size=1,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)

upsampled_logits_batch, fcn_32s_variables_mapping = FCN_32s(image_batch_tensor=image_batch,
                                                           number_of_classes=number_of_classes,
                                                           is_training=True)


valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                     logits_batch_tensor=upsampled_logits_batch,
                                                                                    class_labels=class_labels)



cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                          labels=valid_labels_batch_tensor)

# Normalize the cross entropy -- the number of elements
# is different during each step due to mask out regions
cross_entropy_sum = tf.reduce_mean(cross_entropies)

pred = tf.argmax(upsampled_logits_batch, dimension=3)

probabilities = tf.nn.softmax(upsampled_logits_batch)


with tf.variable_scope("adam_vars"):
    train_step = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cross_entropy_sum)


# Variable's initialization functions
fcn_32s_without_last_variables_mapping = slim.get_variables_to_restore(exclude=['Logits', 'AuxLogits'])
init_fn = slim.assign_from_checkpoint_fn(model_path=inception_v3_checkpoint_path,
                                         var_list=fcn_32s_without_last_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()

tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
     os.makedirs(log_folder)
    
#The op for initializing the variables.
local_vars_init_op = tf.local_variables_initializer()

combined_op = tf.group(local_vars_init_op, global_vars_init_op)

# We need this to save only model variables and omit
# optimization-related and other variables.
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)


with tf.Session()  as sess:
    
    sess.run(combined_op)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # 10 epochs
    for i in xrange(11127 * 10):
    
        cross_entropy, summary_string, _ = sess.run([ cross_entropy_sum,
                                                      merged_summary_op,
                                                      train_step ])
        
        print("Current loss: " + str(cross_entropy))
        
        summary_string_writer.add_summary(summary_string, i)
        
        if i % 11127 == 0:
            save_path = saver.save(sess, "/home/dpakhom1/tf_projects/segmentation/model_fcn32s.ckpt")
            print("Model saved in file: %s" % save_path)
            
        
    coord.request_stop()
    coord.join(threads)
    
    save_path = saver.save(sess, "/home/dpakhom1/tf_projects/segmentation/model_fcn32s.ckpt")
    print("Model saved in file: %s" % save_path)
    
summary_string_writer.close()