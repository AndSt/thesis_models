import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.ops import embedding_ops
from thesis_models.utils import get_standard_binary_estimator_spec


def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
    """
    Computes attentention by first computing an attention score

    Parameters
    ----------
    q: tf.Tensor, shape=[d_1, .., d_r, d_q, d]
        The query.
    k: tf.Tensor, shape=[d_1, .., d_r, d_q, d]
        The keys.
    v: tf.Tensor, shape=[d_1, .., d_r, d_v, d]
        The value tensor. The query tensor
    mode: str
        Either 'general' or 'cosine'. First uses general attention, the second one uses cosine attention.

    Returns
    -------
    attention: tf.Tensor
        The computed vector.
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def mask_tensor(tensor, mask):
    """
    Computes a mask for a given tensor. That means each entry is set to 0.

    Parameters
    ----------
    tensor: tf.Tensor, shape=[d_1, .., d_r, d]
        The input tensor.
    mask: tf.Tensor, shape=[d_1, ...., d_r]
        The masking tensor. It has to hold rank(tensor) = rank(mask) + 1

    Returns
    -------
        masked_tensor: tf.Tensor

    """

    mask = tf.expand_dims(mask, -1)
    mask = tf.broadcast_to(mask, tf.shape(tensor))
    # print(mask.get_shape())
    # print(tensor.get_shape())

    masked_tensor = tf.multiply(tensor, mask)

    return masked_tensor


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads,
        seq_len, depth)
        """
        shape = tf.shape(x)
        last_dim = tf.expand_dims(tf.cast(self.d_model / self.num_heads, tf.int32), 0)
        new_shape = tf.concat([shape[0:-1], tf.expand_dims(self.num_heads, 0), last_dim], 0)

        x = tf.reshape(x, new_shape)
        # print(x.get_shape())
        num_ranks = tf.cast(tf.shape(new_shape)[0], tf.int32)

        val_1 = tf.expand_dims(num_ranks - 1, 0)
        val_2 = tf.expand_dims(num_ranks - 2, 0)
        val_3 = tf.expand_dims(num_ranks - 3, 0)

        perm = tf.range(0, num_ranks - 3)
        perm2 = tf.concat([val_2, val_3, val_1], 0)
        perm = tf.concat([perm, perm2], 0)
        # print(perm2.get_shape())

        return tf.transpose(x, perm=perm), perm

    def call(self, v, k, q):
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q, perm_q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k, perm_k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v, perm_v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        # print(q.get_shape())
        # print(k.get_shape())
        # print(v.get_shape())
        #
        # print(perm_q.get_shape())
        # print(perm_k.get_shape())
        # print(perm_v.get_shape())

        # assert perm_k == perm_v

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=perm_v)  # (batch_size, seq_len_q, num_heads, depth)

        # print('new shape')
        # print(scaled_attention.get_shape())
        new_shape = tf.concat([tf.shape(scaled_attention)[0:-2], tf.expand_dims(self.d_model, 0)], 0)
        concat_attention = tf.reshape(scaled_attention, new_shape)  # (batch_size, seq_len_q, d_model)
        # print(concat_attention.get_shape())

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def path_hier_att_model_fn(features, labels, mode, params):
    # Embeddings of elements: [total_num_elts, dim]
    kg_embeddings = tf.get_variable('kg_embeddings', shape=params['kg_shape'],
                                    initializer=params['kg_initializer'])
    # print(kg_embeddings.get_shape())

    # Mask elements: 0 or 1
    mask_embeddings = tf.get_variable('mask_embeddings', shape=params['mask_shape'],
                                      initializer=params['mask_initializer'])
    # print(mask_embeddings.get_shape())

    # Embeddings of elements: [total_num_elts, dim]
    ext_embeddings = tf.get_variable('ext_embeddings', shape=params['ext_shape'],
                                     initializer=params['ext_initializer'])
    # print(ext_embeddings.get_shape())

    # path: [batch_size, num_paths, max_path_length, node_embedding_dim]
    kg_path_input_layer = embedding_ops.embedding_lookup(kg_embeddings, features['path_elts'])
    # print(kg_path_input_layer.get_shape())
    # print(input_layer.get_shape())
    # 1 because each element is only 0 or 1

    # external entities: [batch_size, node_embedding_dim]
    ext_input_layer = embedding_ops.embedding_lookup(ext_embeddings, features['ext_elts'])
    # print(ext_input_layer.get_shape())

    # path mask: [batch_size, num_paths, max_path_length, 1]
    kg_path_mask = embedding_ops.embedding_lookup(mask_embeddings, features['mask_elts'])
    # print(kg_path_mask.get_shape())

    # path: [batch_size, num_paths, max_path_length, node_embedding_dim]
    kg_path_input_layer = mask_tensor(kg_path_input_layer, kg_path_mask)
    # print(kg_path_input_layer.get_shape())

    with tf.variable_scope('node_level'):
        # shape: [batch_size, 1, nodes_embedding_dim]
        ext_layer = tf.expand_dims(tf.expand_dims(ext_input_layer, 1), 2)
        ext_layer = tf.broadcast_to(ext_layer, [tf.shape(ext_layer)[0], params['num_paths'], 1, params['ext_shape'][1]])
        # print(ext_layer.get_shape())

        # shape: [batch_size, num_paths, 1, embedding_dim]
        node_attention_layer, node_attention_weights = MultiHeadAttention(params['kg_shape'][1], params['num_heads_node_lvl'])(
            kg_path_input_layer, kg_path_input_layer, ext_layer)
        # print(node_attention_layer.get_shape())

        # shape: [batch_size, num_paths, embedding_dim]
        node_attention_layer = tf.squeeze(node_attention_layer, axis=[2])
        # print(node_attention_layer.get_shape())

        node_attention_layer = Dropout(params['dropout'])(node_attention_layer,
                                                          training=mode == tf.estimator.ModeKeys.TRAIN)
        node_attention_layer = Dense(units=node_attention_layer.shape[-1], activation=tf.nn.relu,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                         scale=params['regularization']))(node_attention_layer)

    with tf.variable_scope('path_level'):
        # shape: [batch_size, 1, nodes_embedding_dim]
        ext_layer_2 = tf.expand_dims(ext_input_layer, 1)
        ext_layer_2 = tf.broadcast_to(ext_layer_2, [tf.shape(ext_layer)[0], 1, params['ext_shape'][1]])

        path_attention_layer, path_attention_weights = MultiHeadAttention(params['kg_shape'][1], params['num_heads_path_lvl'])(
            node_attention_layer, node_attention_layer, ext_layer_2)
        # print(path_attention_layer.get_shape())

        # shape: [batch_size, embedding_dim]
        all_paths_attention_layer = tf.squeeze(path_attention_layer, axis=[1])
        all_paths_attention_layer = Dropout(params['dropout'])(all_paths_attention_layer,
                                                               training=mode == tf.estimator.ModeKeys.TRAIN)
        all_paths_attention_layer = Dense(units=all_paths_attention_layer.shape[-1], activation=tf.nn.relu,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                              scale=params['regularization']))(
            all_paths_attention_layer)
        # print(all_paths_attention_layer.get_shape())

    # all_paths_attention_layer = tf.keras.layers.Flatten()(all_paths_attention_layer)
    # print(all_paths_attention_layer.get_shape())
    all_paths_attention_layer = tf.keras.layers.Flatten()(all_paths_attention_layer)
    final_dense = tf.concat([all_paths_attention_layer, ext_input_layer], 1)
    final_dense = Dropout(params['dropout'])(final_dense, training=mode == tf.estimator.ModeKeys.TRAIN)
    # print(final_dense.get_shape())
    final_dense = Dense(units=params['final_dense_units'], activation=tf.nn.relu,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params['regularization']))(
        final_dense)

    # Test additional dense layers:
    # final_dense = Dropout(params['dropout'])(final_dense, training=mode == tf.estimator.ModeKeys.TRAIN)
    # # print(final_dense.get_shape())
    # final_dense = Dense(units=params['final_dense_units'], activation=tf.nn.relu,
    #                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params['regularization']))(
    #     final_dense)
    #
    # final_dense = Dropout(params['dropout'])(final_dense, training=mode == tf.estimator.ModeKeys.TRAIN)
    # # print(final_dense.get_shape())
    # final_dense = Dense(units=params['final_dense_units'], activation=tf.nn.relu,
    #                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params['regularization']))(
    #     final_dense)

    # [batch_size, 1]
    logits = Dense(units=1, activation=None)(final_dense)
    # print(logits.get_shape)


    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    probabilities = tf.nn.sigmoid(logits)
    predicted_classes = tf.round(probabilities)
    probabilities = tf.concat([probabilities, 1 - probabilities], axis=1)
    #print(predicted_classes.get_shape())
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            #'class_ids': tf.squeeze(predicted_classes),
            'probabilities': probabilities,
            'logits': logits,
            'node_attention': node_attention_weights,
            'path_attention': path_attention_weights
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.

    labels = tf.cast(labels, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    precision = tf.metrics.precision(labels=labels, predictions=predicted_classes, name='prec_op')
    recall = tf.metrics.recall(labels=labels, predictions=predicted_classes, name='rec_op')
    f1_score = tf.contrib.metrics.f1_score(labels=labels, predictions=predicted_classes, name='f1_op')
    auc_score = tf.metrics.auc(labels=labels, predictions=predicted_classes, name='auc_op')
    #ap_score = tf.metrics.average_precision_at_k(labels=labels, predictions=predicted_classes, k=10, name='ap_score')

    metrics = {'accuracy': accuracy, 'f1_score': f1_score, 'precision': precision, 'recall': recall,
               'auc': auc_score}#, 'ap_score': ap_score}

    for metric in metrics:
        if metric != 'loss':
            tf.summary.scalar(metric, metrics[metric][1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
