from typing import List
import numpy as np
import random
from bpemb import BPEmb
import tensorflow as tf
import pandas as pd
import os
from typing import Union
import ast


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def binary_search_p(func, ground_truth, test_set, iterations=20):
    """
    Find the best value of p, given the scoring function func
    """
    best_p = -10
    best_func_score = func(ground_truth, test_set >= best_p)
    p_values = np.arange(-1, 2, 0.1)
    for p_value in p_values:
        value = 20 ** p_value

        current_score = func(ground_truth, test_set >= - value)
        if current_score >= best_func_score:
            best_func_score = current_score
            best_p = -value
    return best_p, best_func_score


def subsample_df_by_column(df: pd.DataFrame, column: str, values: List, num_samples_per_value: int) -> pd.DataFrame:
    """
    Takes a DataFrame. For each value in `values` it takes the sub frame which has this value in the column `column`
    and samples `num_samples_per_value` rows.

    Parameters
    ----------
    df: pd.DataFrame
        The pd.DataFrame to look at.
    column: str
        The column, where we search for the values
    values: List
        Typically each value is a user ID and we want a limited sample, so we don't get skewed results.
    num_samples_per_value: int
        The amount of rows to sample for each value
    Returns
    -------
    subsample_df: pd.DataFrame
        The returned DataFrame.
    """
    assert column in df.columns.values
    subsample_df = pd.DataFrame([], columns=df.columns.values)

    for sample_id in values:
        sample_df = df[df[column] == sample_id]
        if sample_df.shape[0] > num_samples_per_value:
            sample_df = sample_df.sample(n=num_samples_per_value, replace=False)
        subsample_df = pd.concat([subsample_df, sample_df])

    return subsample_df.reset_index(drop=True)


def pd_set_minus(df: pd.DataFrame, df_minus: pd.DataFrame) -> pd.DataFrame:
    """
    Definition set minus: Let A, B be sets. Then A - B holds every element which is in A, but not in B
    In that manner we return every row of df, which is not in df_minus.

    Parameters
    ----------
    df: pd.DataFrame
        The original DataFrame.
    df_minus: pd.DataFrame
        The DataFrame who's rows we exclude from df.

    Returns
    -------
    df - df_minus: pd.DataFrame

    """
    df = pd.merge(df, df_minus, how='outer', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
    return df.reset_index(drop=True)


def get_tensorboard_dir(path: str = 'data/models/krpn', prefix: str = 'model') -> str:
    """
    Analyzes the already existing folders in `path` and returns a new directory name to use for tensorboard
    logging.

    Parameters
    ----------
    path: str
        Top level path for all tensorboard outputs
    prefix: str
        String to prefix the models.

    Returns
    -------
    model_dir: str
        A new directory name to log to Tensorboard.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    files = os.listdir(path)
    nums = []
    for file in files:
        nums.append(int(file.replace('{}_'.format(prefix), '')))
    if len(nums) > 0:
        num = np.max(np.array(nums)) + 1
    else:
        num = 0
    model_dir = '{}/{}_{}'.format(path, prefix, num)
    return model_dir


def get_prediction_df(estimator, evaluate_input_fn, data, data_df,
                      columns=['class_ids', 'probabilities', 'logits']):
    """Run all experiments in data and report the prediction results in a
    pd.DataFrame

    Parameters
    ----------
    estimator : tf.estimator
        Previously trained or downloaded Estimator
    evaluate_input_fn :
        Function providing the input to the estimato. See train.py for options.
    data : type
        Description of parameter `data`.
    data_df : pd.DataFrame
        Description of parameter `data_df`.
    columns : type
        Description of parameter `columns`.

    Returns
    -------
    pred_df: pd.DataFrame
        DataFrame holding predcition values and settings for each data split,
        e.g. the train split.

    """

    predictions = estimator.predict(input_fn=lambda: evaluate_input_fn(data[0],data[1], 64))

    preds = []
    i = 0
    for pred_dict in predictions:
        row = [data_df.iloc[i]['kg_uid'], data_df.iloc[i]['ext_uid']]
        for column in columns:
            col = pred_dict[column]
            row.append(col)
        i = i + 1
        preds.append(row)

    columns = ['kg_uid', 'ext_uid'] + columns
    preds_df = pd.DataFrame(preds, columns=columns)
    assert preds_df.shape[0] == data_df.shape[0]
    pred_df = pd.merge(data_df, preds_df, how='left', on=['kg_uid', 'ext_uid'])
    assert pred_df.shape[0] == preds_df.shape[0]

    return pred_df


def add_paths_to_df(df, path_df: Union[str, pd.DataFrame], left_on_col, right_on_col):
    if isinstance(path_df, str):
        path_df = pd.read_csv(path_df, sep=';')
        path_df['paths'] = path_df['paths'].apply(lambda x: eval(x))

    assert isinstance(path_df, pd.DataFrame)
    df = pd.merge(df, path_df, left_on=left_on_col, right_on=right_on_col, how='left').drop(columns=[right_on_col])
    df = df.dropna()
    # df['num_paths'] = df['paths'].apply(lambda x: len(x))
    return df


def add_config_elts_as_cols(row: pd.Series, elts: List[str] = []) -> pd.Series:
    """Takes a row of a pandas DataFrame and creates a new columns which
    hold information from the config dictrionary in row['config'].

    Parameters
    ----------
    row : pd.Series
        Row of a result DataFrame
    elts : List[str]
        Elements of the RunConfig which we want to put into its own column.

    Returns
    -------
    pd.Series
        New row after added columns.

    """

    if type(row['config']) == str:
        config = ast.literal_eval(row['config'])
    else:
        config = row['config']
    for elt in elts:
        if elt in config:
            row[elt] = config[elt]
        else:
            row[elt] = ''
    return row


def cols_to_f1(row):
    add = float(row['precision']) + float(row['recall'])
    mult = 2 * float(row['precision']) * float(row['recall'])
    if add == 0:
        return 0
    return mult / add


def get_standard_binary_estimator_spec(logits, labels, mode, learning_rate):
    """

    Parameters
    ----------
    logits
    labels
    mode
    learning_rate

    Returns
    -------

    """
    # labels is None, if we are in predicition mode
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    probabilities = tf.nn.sigmoid(logits)
    predicted_classes = tf.round(probabilities)
    probabilities = tf.concat([probabilities, 1 - probabilities], axis=1)
    # print(predicted_classes.get_shape())
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': tf.squeeze(predicted_classes),
            'probabilities': probabilities,
            'logits': logits
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

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
