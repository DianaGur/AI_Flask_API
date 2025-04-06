import tensorflow as tf
import numpy as np
import cv2
import os

def load_model(model_dir):
    with tf.Graph().as_default():
        sess = tf.Session()
        meta_file, ckpt_file = get_model_filenames(model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
        saver.restore(sess, os.path.join(model_dir, ckpt_file))
        return sess, tf.get_default_graph()

def get_model_filenames(model_dir):
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.meta'):
            meta_file = file_name
        elif file_name.endswith('.ckpt'):
            ckpt_file = file_name
    return meta_file, ckpt_file

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def load_and_align_image(image_path, image_size=160):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img = prewhiten(img)
    return np.expand_dims(img, axis=0)

def get_embedding(sess, graph, image_path):
    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    image = load_and_align_image(image_path)
    feed_dict = {images_placeholder: image, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding[0]
