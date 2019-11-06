import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.ops import embedding_ops

from thesis_models.utils import get_standard_binary_estimator_spec


def krpn_model_fct(features, labels, mode, params):

    # [num_elts, embedding_dim]
    elt_embeddings = tf.get_variable('elts_embeddings', shape=params['elts_shape'],
                                     initializer=params['elts_initializer'])

    # path: [batch_size, num_paths, max_path_length, embedding_dim]
    input_layer = embedding_ops.embedding_lookup(elt_embeddings, features['path_elts'])
    # print(input_layer.get_shape())
    # 1 because each element is only 0 or 1

    # path: [0, 1]
    mask_embeddings = tf.get_variable('mask_embeddings', shape=params['mask_shape'],
                                      initializer=params['mask_initializer'])

    # path mask: [batch_size, num_paths, max_path_length, 1]
    mask = embedding_ops.embedding_lookup(mask_embeddings, features['mask_elts'])
    # print(mask.get_shape())

    # path: [batch_size * num_paths,  max_path_length, node_embedding_dim]
    lstm_input = tf.reshape(input_layer, shape=(-1, params['path_length'], params['elts_embedding_dim']))
    # print(lstm_input.get_shape())

    # masks: [batch_size * num_paths, max_path_length]
    mask_input = tf.reshape(mask, shape=(-1, params['path_length']))
    # print(mask_input.get_shape())

    regularizer = tf.contrib.layers.l2_regularizer(scale=params['regularization'])
    # [batch_size * num_paths, lstm_dim]
    lstm_layer = LSTM(units=params['lstm_units'], kernel_regularizer=regularizer)(lstm_input, mask=mask_input)
    # print(lstm_layer.get_shape())

    # [batch_size * num_paths, dense_1_dim]
    dense_1 = Dense(units=params['dense_units_1'], activation='relu')(lstm_layer)

    # [batch_size * num_paths, dense_1_dim]
    droupout_1 = Dropout(rate=params['dropout'])(dense_1, training=mode == tf.estimator.ModeKeys.TRAIN)

    # [batch_size * num_paths, dense_2_dim]
    dense_2 = Dense(units=params['dense_units_2'])(droupout_1)
    # print(dense_2.get_shape())

    dense_2 = tf.exp(dense_2 / params['gamma'])

    # [batch_size, num_paths, dense_2_dim]
    pre_weight_pooling = tf.reshape(dense_2, shape=(-1, params['num_paths'],
                                                    params['dense_units_2']))
    # print(pre_weight_pooling.get_shape())

    # [batch_size, dense_2_dim]
    weight_pooling = tf.log(tf.reduce_sum(pre_weight_pooling, axis=[1]))
    # print(weight_pooling.get_shape())

    droupout_2 = Dropout(params['dropout'])(weight_pooling, training= mode == tf.estimator.ModeKeys.TRAIN)

    # [batch_size, 1]
    logits = Dense(units=1, activation=None)(droupout_2)

    return get_standard_binary_estimator_spec(logits, labels, mode, params['learning_rate'])
