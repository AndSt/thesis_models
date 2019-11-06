from typing import List, Dict
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import uuid
import itertools as it
from thesis_models.config import checkpoint_config
from thesis_models.utils import get_tensorboard_dir


def train_input_fn(features, labels, batch_size):
    """An input function for training"""

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset


def evaluate_input_fn(features, labels, batch_size=100):
    """An input function for evaluation"""
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset


def predict_input_fn(features, batch_size=100):
    """An input function for evaluation"""
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset


def get_sim_model_params(features, X_kg, X_ext, config):

    my_feature_columns = []
    for key in features.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key, shape=features[key].shape))

    def kg_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return X_kg

    def ext_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return X_ext

    params = {
        'dense_units_1': config['dense_units_1'] if config['dense_units_1'] else 128,
        'dense_units_2': config['dense_units_2'] if config['dense_units_2'] else 64,
        'dropout_rate': config['dropout_rate'] if config['dropout_rate'] else 64,
        'learning_rate': config['learning_rate'] if config['learning_rate'] else 0.01,
        'feature_columns': my_feature_columns,
        'kg_initializer': kg_initializer,
        'kg_size': X_kg.shape[0],
        'kg_embedding_dim': X_kg.shape[1],
        'ext_initializer': ext_initializer,
        'ext_size': X_ext.shape[0],
        'ext_embedding_dim': X_ext.shape[1]
    }

    return params


def get_gcn_model_params(features, X_kg, X_ext, A, config=None):
    my_feature_columns = []
    for key in features.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key, shape=features[key].shape))

    def kg_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return X_kg

    def kg_elt_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return np.arange(0, X_kg.shape[0])

    def ext_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return X_ext

    params = {
        'kg_units_1': config['kg_units_1'] if 'kg_units_1' in config else 256,
        'kg_units_2': config['kg_units_2'] if 'kg_units_2' in config else 256,
        'kg_output_units': config['kg_output_units'] if 'kg_output_units' in config else 128,
        'ext_output_units': config['ext_output_units'] if 'ext_output_units' in config else 256,
        'concat_output_units': config['concat_output_units'] if 'concat_output_units' in config else 128,
        'learning_rate': config['learning_rate'] if 'learning_rate' in config else 0.01,
        'adjacency_matrix': A,
        'feature_columns': my_feature_columns,
        'kg_initializer': kg_initializer,
        'kg_elt_initializer': kg_elt_initializer,
        'kg_size': X_kg.shape[0],
        'kg_embedding_dim': X_kg.shape[1],
        'ext_initializer': ext_initializer,
        'ext_size': X_ext.shape[0],
        'ext_embedding_dim': X_ext.shape[1]
    }

    return params


def get_krpn_model_params(X, features, num_paths, path_length, config=None):

    my_feature_columns = []
    for key in features.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key, shape=features[key].shape))

    def elts_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return X

    mask_elts = np.array([0, 1])

    def mask_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return mask_elts

    params = {
        'lstm_units': config['lstm_units'] if 'lstm_units' in config else 256,
        'dense_units_1': config['dense_units_1'] if config['dense_units_1'] else 64,
        'dense_units_2': config['dense_units_2'] if config['dense_units_2'] else 64,
        'learning_rate': config['learning_rate'] if 'learning_rate' in config else 0.01,
        'feature_columns': my_feature_columns,
        'num_paths': num_paths,
        'path_length': path_length,
        'gamma': 1.0,
        'dropout': config['dropout'] if 'dropout' in config else 0.5,
        'regularization': config['regularization'] if 'regularization' in config else 0.0,
        'elts_initializer': elts_initializer,
        'elts_embedding_dim': X.shape[1],
        'elts_shape': X.shape,
        'mask_initializer': mask_initializer,
        'mask_shape': mask_elts.shape,
    }

    return params


def get_path_hier_att_model_params(X_kg, X_ext, features, num_paths, path_length, config=None):

    my_feature_columns = []
    for key in features.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key, shape=features[key].shape))

    def kg_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return X_kg

    def ext_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return X_ext

    mask_elts = np.array([0, 1])

    def mask_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return mask_elts

    params = {
        'final_dense_units': config['final_dense_units'] if 'final_dense_units' in config else 64,
        'learning_rate': config['learning_rate'] if 'learning_rate' in config else 0.005,
        'feature_columns': my_feature_columns,
        'num_paths': num_paths,
        'path_length': path_length,
        'mode': config['mode'] if 'mode' in config else 'general', # takes 'cosine', 'general' and 'lstm'
        'num_heads_node_lvl': config['num_heads_node_lvl'] if 'num_heads_node_lvl' in config else 1,
        'num_heads_path_lvl': config['num_heads_path_lvl'] if 'num_heads_path_lvl' in config else 1,
        'dropout': config['dropout'] if 'dropout' in config else 0.5,
        'regularization': config['regularization'] if 'regularization' in config else 0.0,
        'kg_initializer': tf.constant_initializer(X_kg),
        'kg_embedding_dim': X_kg.shape[1],
        'kg_shape': X_kg.shape,
        'ext_initializer': tf.constant_initializer(X_ext),
        'ext_embedding_dim': X_ext.shape[1],
        'ext_shape': X_ext.shape,
        'mask_initializer': mask_initializer,
        'mask_shape': mask_elts.shape,
    }

    return params


def evaluate_model(estimator, input_data, results_dir: str = None, config: Dict = None, metrics: List[str] = [],
                   tracked_config_elts: List[str]= ['model_name', 'max_steps', 'batch_size']) -> pd.DataFrame:
    """
    Takes a fully trained estimator model, evaluates all data splits in `input_data` and saves the wanted data in
    a pd.DataFrame

    Parameters
    ----------
    estimator:

    input_data:
        The input data for the model. Contains splits and per split a feature and labelarray.
    results_dir: str
        The dictionary where the result is stored
    config: Dict
        A dictionary describing the model run
    metrics: List[str]
        List of metrics to keep track of. For instance: 'accuracy'
    tracked_config_elts: List[str]
        The subset of the configuration dict of the model run to track.

    Returns
    -------
    results_df: pd.DataFrame
        We concat the new results to the previous results and return the new DataFrame
    """

    # Prepare the needed data fields
    if len(metrics) == 0:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']

    results_df = []
    columns = ['run_id', 'split name'] + metrics + ['config']
    run_id = uuid.uuid4()

    # run model on data splits and store data in a DataFrame
    for val in input_data:
        result_row = [run_id, val]
        ev = estimator.evaluate(input_fn=lambda: evaluate_input_fn(input_data[val][0], input_data[val][1], config['batch_size']))
        for metric in metrics:
            if metric in ev:
                result_row.append(ev[metric])
            else:
                result_row.append('')
        if config:
            subconfig = {k: config[k] for k in config.keys() & tracked_config_elts}
            result_row.append(subconfig)
        else:
            result_row.append('')
        results_df.append(result_row)

    results_df = pd.DataFrame(results_df, columns=columns)

    # save the DataFrame and possibly concat it to already exisiting DataFrame
    if results_dir:
        if os.path.exists('{}/results.csv'.format(results_dir)):
            old_results_df = pd.read_csv('{}/results.csv'.format(results_dir), sep=';')
            results_df = pd.concat([old_results_df, results_df], ignore_index=True, sort=False)

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        results_df.to_csv('{}/results.csv'.format(results_dir), index=False, sep=';')

    return results_df


def run_train_and_evaluate(data, config: Dict, results_dir: str = None, tracked_config_elts: List[str] = ['model_name', 'max_steps', 'batch_size']):
    """

    Parameters
    ----------
    data: Dict[str, Tuple[Dict[str, np.array], np.array]]
        First str is the data split, e.g. 'train'. Tuple contains, features and labels. For each label there is np.array
        per dict str. %TODO better description
    config: Dict
        Dictionary containing the configuration for a model
    results_dir: str
        Directory where the evaluation results are stored
    tracked_config_elts: List[str]
        List of configuration elements which should be kept track of

    Returns
    -------
    estimator: tf.estimator.Estimator
        A trained estimator.
    results_df: pd.DataFrame
        DataFrame containing the evaluation results for this run and all other results stored in `results_dir`
    """
    estimator = tf.estimator.Estimator(
       model_fn=config['model_fn'],
       #model_dir=config['model_dir'], # disabled as model save dir got too huge given hyperparameter search.
       params=config['params'],
       config=config['checkpoint_config']
    )

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(data['train'][0], data['train'][1], \
                                                            config['batch_size']), max_steps=config['max_steps'])

    eval_spec = tf.estimator.EvalSpec(
       input_fn=lambda: evaluate_input_fn(data['val'][0], data['val'][1], config['batch_size']),
       steps=1,
       start_delay_secs=1,
       throttle_secs=config['eval_throttle_secs'] # evaluate every 10 seconds
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    results_df = evaluate_model(estimator, data, config=config, results_dir=results_dir, tracked_config_elts=tracked_config_elts)

    return estimator, results_df


def run_hyp_param_run(model_fct, input_data, hyp_param_search_space, param_fct, param_fct_args, tracked_config_elts,
                      data_dir='data/', gpu=1):

    all_run_hyp_params = [dict(zip(hyp_param_search_space.keys(), a)) for a in
                          list(it.product(*hyp_param_search_space.values()))]

    for hyp_params in all_run_hyp_params:
        run_config = {
            'model_name': 'sim_bpe',
            'eval_throttle_secs': 15,
            'checkpoint_config': checkpoint_config(gpu=gpu),
            'model_fn': model_fct
        }
        run_config.update(hyp_params)
        run_config['params'] = param_fct(*param_fct_args, run_config)
        run_config['model_dir'] = get_tensorboard_dir('data/models/{}'.format(run_config['model_name']), 'model')

        estimator, result_df = run_train_and_evaluate(input_data, run_config,
                                                      results_dir='{}/results'.format(data_dir),
                                                      tracked_config_elts=tracked_config_elts)

    return estimator, result_df