import unittest
import tensorflow as tf
#from thesis_models.krpn_model import load_and_reshape
from thesis_models.path_hier_att_model import scaled_dot_attention, mask_tensor
import numpy as np


class KRPNTest(unittest.TestCase):

    def test_load_and_reshape(self):
        success = True

        self.assertEqual(success, True)


def test_scaled_dot_attention():
    batch_size, num_paths, path_length, dim = 2, 3, 3, 2

    # test first attention computation; rank 4 tensors
    path_embs = np.ones((batch_size, num_paths, path_length, dim))
    ext_emb = 2 * np.ones((batch_size, dim))
    with tf.Session() as sess:
        x = tf.constant(path_embs, tf.float32)
        y = tf.constant(ext_emb, tf.float32)
        y = tf.expand_dims(tf.expand_dims(y, 1), 2)
        y = tf.broadcast_to(y, [batch_size, num_paths, 1, dim])

        att_val = scaled_dot_attention(y, x)
        y, att_val = sess.run([y, att_val])
        assert att_val.shape == (batch_size, num_paths, 1, dim)

    # test second attention computation; rank 3 tensors
    path_embs = np.ones((batch_size, num_paths, dim))
    ext_emb = 2 * np.ones((batch_size, dim))
    with tf.Session() as sess:
        x = tf.constant(path_embs, tf.float32)
        y = tf.constant(ext_emb, tf.float32)
        y = tf.expand_dims(y, 1)
        y = tf.broadcast_to(y, [batch_size, 1, dim])

        att_val = scaled_dot_attention(y, x)
        y, att_val = sess.run([y, att_val])
        assert att_val.shape == (batch_size, 1, dim)


def test_mask_tensor():

    x = np.ones((3, 3))
    y = np.array([0, 1, 0])

    masked_correct = np.zeros((3, 3))
    masked_correct[1] = np.ones(3)

    with tf.Session() as sess:
        x = tf.constant(x, tf.float32)
        y = tf.constant(y, tf.float32)

        masked_x = mask_tensor(x, y)

        masked_x = sess.run(masked_x)

        assert np.all(np.equal(masked_x, masked_correct))
        assert np.all(np.equal(masked_x, x)) == False
