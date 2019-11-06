import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from thesis_models.utils import get_standard_binary_estimator_spec


def sim_model_fn(features, labels, mode, params):

    # [batch_size, 1, graph_size]
    graph_input_layer = tf.contrib.layers.embed_sequence(features['graph_elts'], params['kg_size'], 1,
                                                         initializer=params['kg_initializer'])

    # [batch_size, 1, text_embedding_dim]
    text_input_layer = tf.contrib.layers.embed_sequence(
        features['text_elts'], params['ext_size'], params['ext_embedding_dim'],
        initializer=params['ext_initializer'])

    # [batch_size, 1 , graph_size + text_embedding_dim]
    concat = tf.concat([graph_input_layer, text_input_layer], axis=2)
    # [batch_size, graph_size + text_embedding_dim]
    concat = tf.squeeze(concat, [1])

    # [batch_size, linear_units_1]
    net_1 = Dense(units=params['dense_units_1'], activation=tf.nn.relu)(concat)
    # [batch_size, linear_units_1]
    net_1 = Dropout(rate=params['dropout_rate'])(net_1, training= mode == tf.estimator.ModeKeys.TRAIN)

    # [batch_size, linear_units_2]
    net_2 = Dense(units=params['dense_units_2'], activation=tf.nn.relu)(net_1)
    # [batch_size, linear_units_2]
    net_2 = Dropout(rate=params['dropout_rate'])(net_2, training= mode == tf.estimator.ModeKeys.TRAIN)

    # [batch_size, 1]
    logits = Dense(units=1, activation=None)(net_2)

    return get_standard_binary_estimator_spec(logits, labels, mode, params['learning_rate'])
