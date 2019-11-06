import tensorflow as tf
from tensorflow.python.keras import initializers, activations
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from thesis_models.utils import convert_sparse_matrix_to_sparse_tensor, \
    get_standard_binary_estimator_spec


class GraphConvolutionSparse(tf.keras.layers.Layer):
    """Graph convolution layer for sparse inputs"""

    def __init__(self, units, input_dim, adj, dropout=0.,
                 kernel_initializer='glorot_uniform', use_bias=False,
                 bias_initializer='zeros', act=tf.nn.relu, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        self.input_dim = input_dim

        # TODO look at VAE paper
        self.graph_units_1 = 64
        self.graph_units_2 = 64

        self.units = units
        self.dropout = dropout
        self.adj = adj

        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.act = act
        self.issparse = True

    def compute_output_shape(self, input_shapes):
        output_shape = (self.input_dim, self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        x = tf.matmul(x, self.kernel)
        x = tf.sparse_tensor_dense_matmul(self.adj, x)

        if self.bias:
            x += self.bias
        outputs = self.act(x)
        return outputs

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }

        base_config = super(GraphConvolutionSparse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def gcn_model_fn(features, labels, mode, params):
    #######################
    # Compute graph layer #
    #######################

    # [batch_size, 1]
    graph_elts_layer = tf.contrib.layers.embed_sequence(features['graph_elts'],
                                                        params['kg_size'], 1, initializer=params['kg_elt_initializer'])
    graph_elts_layer = tf.cast(graph_elts_layer, dtype=tf.int32)
    # print(graph_elts_layer.get_shape())

    # [graph_size, graph_size]
    adjacency_matrix = convert_sparse_matrix_to_sparse_tensor(params['adjacency_matrix'])
    adjacency_matrix = tf.cast(adjacency_matrix, dtype=tf.float32)
    # print(adjacency_matrix.get_shape())

    # [graph_size, embedding_dim]
    graph_input_layer = tf.get_variable('X_kg', shape=[params['kg_size'], params['kg_embedding_dim']],
                                        initializer=params['kg_initializer'], trainable=False)
    # print(graph_input_layer.get_shape())

    # [graph_size, graph_units_1]
    graph_layer_1 = GraphConvolutionSparse(units=params['kg_units_1'], input_dim=params['kg_embedding_dim'],
                                           use_bias=False, adj=adjacency_matrix)(graph_input_layer)

    # TODO sparse dropout

    # [graph_size, graph_units_2]
    graph_layer_2 = GraphConvolutionSparse(units=params['kg_units_2'], input_dim=params['kg_units_1'],
                                           use_bias=False, adj=adjacency_matrix)(graph_layer_1)

    # [batch_size, graph_units_2]
    graph_output = tf.gather_nd(graph_layer_2, graph_elts_layer)
    # print(graph_output.get_shape())

    graph_output = Dense(units=params['kg_output_units'], activation=tf.nn.relu)(graph_output)

    # [batch_size, net_1_units]
    net_1 = Dropout(rate=0.5)(graph_output, training=mode == tf.estimator.ModeKeys.TRAIN)

    ##############################
    #### Compute textual part ####
    ##############################

    # [batch_size, 1, embedding_dim]
    text_input_layer = tf.contrib.layers.embed_sequence(features['text_elts'], params['ext_size'],
                                                        params['ext_embedding_dim'],
                                                        initializer=params['ext_initializer'])
    # print(text_input_layer.get_shape())

    # [batch_size, 1, units_text]
    net_2 = Dropout(rate=0.5)(Dense(units=params['ext_output_units'], activation=tf.nn.relu)(text_input_layer),
                              training=mode == tf.estimator.ModeKeys.TRAIN)
    # [batch_size, units_text]
    net_2 = tf.squeeze(net_2, [1])

    net_2 = Dense(units=params['ext_output_units'], activation=tf.nn.relu)(net_2)

    ####################
    #### Merge them ####
    ####################

    # Combine graph and textual features
    # [batch_size, graph_units_1 + units_text]
    net = Flatten()(tf.concat([net_1, net_2], axis=1))
    net = Dropout(rate=0.5)(net, training=mode == tf.estimator.ModeKeys.TRAIN)
    net = Dense(units=params['concat_output_units'], activation=tf.nn.relu)(net)
    net = Dense(units=params['concat_output_units'], activation=tf.nn.relu)(net)

    # [batch_size, 1]
    logits = Dense(units=1)(net)

    return get_standard_binary_estimator_spec(logits, labels, mode, params['learning_rate'])
