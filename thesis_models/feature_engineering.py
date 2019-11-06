from typing import List, Dict, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import random
from thesis_models.utils import subsample_df_by_column, pd_set_minus


def sample_test_df(df: pd.DataFrame, uid_column: str, train_uids: List, test_uids: List, samples_per_uid: int) \
        -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subsamples rows from `df`. Sampled are `samples_per_uid` for each uid occuring in `train_uids` or `test_uids`.
    The reason is that we want to differentiate between ID's occuring during training and new ID's as these are
    completely new to the model.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to look at.
    uid_column: str
        The column containing the uid's we look for.
    train_uids: List
        A list of ID's the model knows from training
    test_uids: List
        A list of ID's which are new to the model
    samples_per_uid: int
        The amount of sampled rows for each ID. We set this to be equal for both types of ID's to better compare them.

    Returns
    -------
    test_unique_uids_df: pd.DataFrame

    test_train_uids_df: pd.DataFrame

    unassigned_df: pd.DataFrame
        A DataFrame containing all rows in `df` which don't occur in the two DataFrames above
    """

    test_unique_uids_df = subsample_df_by_column(df, uid_column, test_uids, samples_per_uid)
    unassigned_df = pd_set_minus(df, test_unique_uids_df)

    test_train_uids_df = subsample_df_by_column(unassigned_df, uid_column, train_uids, samples_per_uid)
    unassigned_df = pd_set_minus(unassigned_df, test_train_uids_df)

    test_df = pd.concat([test_unique_uids_df, test_train_uids_df], ignore_index=True)

    assert test_df.shape[0] == test_df.drop_duplicates().shape[0]
    assert test_df.shape[0] == test_train_uids_df.shape[0] + test_unique_uids_df.shape[0]

    return test_unique_uids_df, test_train_uids_df, unassigned_df


def train_val_test_split(elts: List, val_test_ratio: float = 0.2) -> [List, List, List]:
    """
    Given a list, we return a train, validation and test split.

    Parameters
    ----------
    elts: List
        List of elements of arbitrary type. Set minus on the List must be supported.
    val_test_ratio: float(0,1)
        If for instance = 0.1, 80% of the elements will be in the train split, 10% in the validation and 10% in the test
        set.

    Returns
    -------
    train_elts, val_elts, test_elts: [List, List, List]
    """

    assert val_test_ratio >= 0
    assert val_test_ratio <= 1

    val_test_ratio = int(len(elts) * val_test_ratio)

    test_elts = set(random.sample(elts, val_test_ratio))
    train_val_elts = elts - test_elts

    val_elts = set(random.sample(train_val_elts, val_test_ratio))
    train_elts = train_val_elts - set(val_elts)

    train_elts = list(train_elts)
    val_elts = list(val_elts)
    test_elts = list(test_elts)

    assert len(train_elts) == len(set(train_elts))
    assert len(val_elts) == len(set(val_elts))
    assert len(test_elts) == len(set(test_elts))
    assert len(elts) == len(train_elts + val_elts + test_elts)

    return train_elts, val_elts, test_elts


def entity_to_feature_row_dict(entities: List, prefix: str = ''):
    """Relates entities to its rumber in the feature matrix.

    Parameters
    ----------
    entities : List[str]
        A list of entities, named arbitrarily. Usually serve as ID to a
        different data structure.
    prefix : str
        Prefix in the dictionary. Helpful if dict is merged with another dict.

    Returns
    -------
    entity_to_feature_row_dict: Dict[str, int]
        Relates a feature row to an identiy
    feature_row_to_entity_dict: Dict[int, str]
        Relates an entity to the corresponding feature.
    i: int
        Total number feature rows.

    """
    entity_to_feature_row_dict = {}
    feature_row_to_entity_dict: Dict = {}
    i = 0
    for entity in entities:
        if prefix == '':
            entity = '{}'.format(entity)
        else:
            entity = '{}_{}'.format(prefix, entity)
        feature_row_to_entity_dict[i] = entity
        entity_to_feature_row_dict[entity] = i
        i = i + 1

    return entity_to_feature_row_dict, feature_row_to_entity_dict, i


def sample_negatives(df, pos_df, pos_neg_ratio):
    ## first negative user - item
    ## then
    unique_users = list(set(pos_df['user_uid']))

    samples_df = []

    for index, row in pos_df.iterrows():

        temp_df = df[df['UID'] == row['user_uid']]
        assert temp_df.shape[0] == 1
        temp = temp_df.iloc[0]

        temp_df = df[df['UID'] != row['user_uid']]
        temp_df = temp_df[temp_df['label'] == temp['label']]

        temp_sample_df = temp_df.iloc[np.random.choice(temp_df.shape[0], pos_neg_ratio, replace=False)].copy()
        # print(temp_sample_df)
        for ind2, row2 in temp_sample_df.iterrows():
            add_row = row.copy()
            add_row[0] = row2['UID']
            samples_df.append(add_row)

        temp_df = pos_df[pos_df['item_uid'] != row['item_uid']]

        temp_sample_df = temp_df.iloc[np.random.choice(temp_df.shape[0], pos_neg_ratio, replace=False)].copy()
        # print(temp_sample_df)
        for ind2, row2 in temp_sample_df.iterrows():
            add_row = row2.copy()
            add_row[1] = row['item_uid']
            samples_df.append(add_row)

    samples_df = pd.DataFrame(samples_df, columns=pos_df.columns.values)

    return samples_df


def get_split(df: pd.DataFrame, num_elts, val_test_size: float = 0.15) -> List[pd.DataFrame]:
    """

    Parameters
    ----------
    df: pd.DataFrame
    num_elts: pd.DataFrame
    val_test_size: float

    Returns
    -------
    [train_df, val_df, test_df]: List[pd.DataFrame]

    """
    # make sure we don't sample more elements as exist
    if num_elts > df.shape[0]:
        num_elts = df.shape[0]

    df = df.reset_index(drop=True)

    # which units do we want to use
    used_elts = np.random.choice(df.shape[0], num_elts, replace=False)
    # absolute size of validation/test set
    val_test_size = int(val_test_size * num_elts)

    train_val_set, test_set = train_test_split(used_elts, test_size=val_test_size)
    train_set, val_set = train_test_split(train_val_set, test_size=val_test_size)

    assert train_set.shape[0] == np.unique(train_set).shape[0]
    assert val_set.shape[0] == np.unique(val_set).shape[0]
    assert test_set.shape[0] == np.unique(test_set).shape[0]

    return df.iloc[train_set], df.iloc[val_set], df.iloc[test_set]


def get_train_val_test_split(pos_df: pd.DataFrame, neg_df: pd.DataFrame, num_pos: int = None, pos_neg_ratio: int = 2,
                             val_test_size: float = 0.15):
    """

    Parameters
    ----------
    pos_df
    neg_df
    pos_elts
    pos_neg_ratio
        If == k, then there are k times as many negative examples

    val_test_size

    Returns
    -------

    """
    pos_df['label'] = 1
    neg_df['label'] = 0
    num_neg = num_pos * pos_neg_ratio
    # first check whether we don't sample too many negative samples
    # then check whether we don't sample too many positive samples
    if num_neg > neg_df.shape[0]:
        num_neg = neg_df.shape[0]
        num_pos = int(pos_df.shape[0] / pos_neg_ratio)
        if num_pos > pos_df.shape[0]:
            num_pos = pos_df.shape[0]
            num_neg = int(num_pos * pos_neg_ratio)

    pos_train_df, pos_val_df, pos_test_df = get_split(pos_df, num_pos, val_test_size)
    neg_train_df, neg_val_df, neg_test_df = get_split(neg_df, num_neg, val_test_size)

    train_df = pd.concat([pos_train_df, neg_train_df], ignore_index=True)
    val_df = pd.concat([pos_val_df, neg_val_df], ignore_index=True)
    test_df = pd.concat([pos_test_df, neg_test_df], ignore_index=True)

    assert val_df.shape == test_df.shape
    assert (train_df.shape[0] + val_df.shape[0] + test_df.shape[0]) == num_pos + num_neg

    return train_df, val_df, test_df


def get_entity_dicts(kg_uids: List[Union[int, str]], external_uids: List[Union[int, str]], kg_prefix: str = 'kg',
                     external_prefix: str = 'ext') -> [Dict[int, str], Dict[str, int], int]:
    """
    This function maps all entities used in the model to a common dataspace. Main application is that they are
    used in the same feature matrix. Therefore position %TODO finish commenting; where to introduce kg and external naming


    Parameters
    ----------
    kg_uids: List[Union[int, str]]
        The knowledge graph entities.
    external_uids: List[Union[int, str]]
        The external entities.
    kg_prefix: str
        String to prefix the knowledge graph ID.
    external_prefix: str
        String to prefix the external ID.

    Returns
    -------
    feature_row_to_entity_dict: Dict[int, str]
    entity_to_feature_row_dict: Dict[str, int]
    num_features: int
    """

    feature_row_to_entity_dict = {}
    entity_to_feature_row_dict = {}
    num_features = 0
    for kg_uid in kg_uids:
        kg_uid = '{}_{}'.format(kg_prefix, kg_uid)
        feature_row_to_entity_dict[num_features] = kg_uid
        entity_to_feature_row_dict[kg_uid] = num_features
        num_features = num_features + 1

    for external_uid in external_uids:
        external_uid = '{}_{}'.format(external_prefix, external_uid)
        feature_row_to_entity_dict[num_features] = external_uid
        entity_to_feature_row_dict[external_uid] = num_features
        num_features = num_features + 1

    return feature_row_to_entity_dict, entity_to_feature_row_dict, num_features


def samples_df_to_entity_dicts(df: pd.DataFrame) -> [Dict[int, str], Dict[str, int], int]:
    """
    Takes a data frame, containing kg entities and returns
    %TODO comment

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    feature_row_to_entity_dict: Dict[int, str]
    entity_to_feature_row_dict: Dict[str, int]
    num_features: int
    """
    assert 'ext_uid' in df.columns.values
    assert 'paths' in df.columns.values

    external_uids = df['ext_uid'].unique().tolist()

    # first each row entry is a list of list. After looking at all rows, we have another list of lists.
    df['paths'] = df['paths'].apply(lambda x: [item for sublist in x for item in sublist])
    kg_uids = list(set([item for sublist in df['paths'].tolist() for item in sublist]))
    return get_entity_dicts(kg_uids, external_uids)


def df_to_tf_input(df: pd.DataFrame, num_paths, path_length):
    """
    Takes paths and pads and masks them.
    We denote the amount of elements as N=df.shape[0]

    Parameters
    ----------
    df: pd.DataFrame
        Has to hold a column `feature_path` where the paths are saved as a list of list.(List[List[int]])
    num_paths: int
        The amount of paths per user-item relation
    path_length: int
        The maximum length of a path.

    Returns
    -------
    path_features: np.array
        Holds an array of IDs of shape (N, num_paths, path_length)
    mask_features: np.array
        The corresponding mask array to path_features (N, num_paths, path_length)
    """
    assert 'feature_paths' in df.columns.values
    assert 'label' in df.columns.values

    path_features = []
    mask_features = []
    labels = []
    empty_path = [[0] * path_length]

    for idx, row in df.iterrows():
        paths = row['feature_paths']

        path_feature = []
        mask_feature = []
        # pad and mask each path
        for path in paths:
            mask = [1] * len(path) + [0] * (path_length - len(path))
            mask_feature.append(mask)

            path = path + (path_length - len(path)) * [0]
            path_feature.append(path)

        # pad and mask the amount of paths per user-item relation
        mask_feature = mask_feature + empty_path * (num_paths - len(path_feature))
        path_feature = path_feature + empty_path * (num_paths - len(path_feature))

        path_features.append(path_feature)
        mask_features.append(mask_feature)
        labels.append(row['label'])

    path_features = np.array(path_features, dtype=np.int)
    mask_features = np.array(mask_features, dtype=np.int)

    assert np.max(mask_features) <= 1 and np.min(mask_features) >= 0
    assert path_features.shape == mask_features.shape
    assert path_features.shape[1:] == (num_paths, path_length)

    features = {'path_elts': path_features, 'mask_elts': mask_features}
    labels = np.array(labels, dtype=np.int)

    return [features, labels]


def paths_to_tf_input(feature_paths: pd.Series, num_paths, path_length):
    """
    Takes paths and pads and masks them.
    We denote the amount of elements as N=df.shape[0]

    Parameters
    ----------
    feature_paths: pd.DataFrame
        Has to hold a column `feature_path` where the paths are saved as a list of list.(List[List[int]])
    num_paths: int
        The amount of paths per user-item relation
    path_length: int
        The maximum length of a path.

    Returns
    -------
    path_features: np.array
        Holds an array of IDs of shape (N, num_paths, path_length)
    mask_features: np.array
        The corresponding mask array to path_features (N, num_paths, path_length)
    """
    assert type(feature_paths) == pd.Series
    path_features = []
    mask_features = []
    empty_path = [[0] * path_length]

    for idx, paths in feature_paths.iteritems():
        assert type(paths) == list
        # print(paths)
        path_feature = []
        mask_feature = []
        # pad and mask each path
        for path in paths:
            mask = [1] * len(path) + [0] * (path_length - len(path))
            mask_feature.append(mask)

            path = path + (path_length - len(path)) * [0]
            path_feature.append(path)

        # pad and mask the amount of paths per user-item relation
        mask_feature = mask_feature + empty_path * (num_paths - len(path_feature))
        path_feature = path_feature + empty_path * (num_paths - len(path_feature))
        # print(len(mask_feature))
        # print(len(path_feature))
        assert len(mask_feature) == len(path_feature)
        assert len(mask_feature) == num_paths

        path_features.append(path_feature)
        mask_features.append(mask_feature)
    # print(path_features)
    path_features = np.array(path_features, dtype=np.int)
    mask_features = np.array(mask_features, dtype=np.int)

    assert np.max(mask_features) <= 1 and np.min(mask_features) >= 0
    assert path_features.shape == mask_features.shape
    assert path_features.shape[1:] == (num_paths, path_length)

    features = {'path_elts': path_features, 'mask_elts': mask_features}
    return features


def xavier_initialization(num_elements: int, dim: int, factor: int = 2) -> np.array:
    """
    Xavier initialization.
    Paper:
    @InProceedings{pmlr-v9-glorot10a,
      title = 	 {Understanding the difficulty of training deep feedforward neural networks},
      author = 	 {Xavier Glorot and Yoshua Bengio},
      pdf = 	 {http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf},
      url = 	 {http://proceedings.mlr.press/v9/glorot10a.html}
      }
    Implementation was inspired by tensorflow:
        https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/layers/python/layers/initializers.py

    Parameters
    ----------
    num_elements: int
        Number of elements/rows the data should have
    dim: int
        Dimension of each feature vector
    factor: int
        Scaling factor. Descriptions are given in the paper or the tensorflow documentation.

    Returns
    -------
    X: np.array
        uniformely distributen matrix of shape (num_elements, dim)
    """
    n = (num_elements + dim) / 2
    limit = math.sqrt(3.0 * factor / n)
    X = np.random.uniform(-limit, limit, (num_elements, dim))
    return X


def path_col_to_feature_path_col(paths: pd.Series, entity_to_feature_row_dict, entitiy_prefix='kg'):
    """"""
    new_paths = []
    for path in paths:
        new_path = []
        # first add user_uid and then all movie_uids of the path

        for node in path:
            new_path.append(entity_to_feature_row_dict['{}_{}'.format(entitiy_prefix, node)])
        new_paths.append(new_path)

    return new_paths


def normalize_feature_path_col(x, max_path_length, max_paths):
    assert type(x) == list
    x = [path for path in x if len(path) <= max_path_length]
    if len(x) < max_paths:
        return x
    x = random.sample(x, max_paths)
    return x


def ext_col_to_feature_path_col(row: pd.Series, entity_to_feature_row_dict, mode: str = '', ext_col_name='ext_uid',
                                ext_prefix='ext'):
    assert 'feature_paths' in row
    assert ext_col_name in row

    new_paths = list()
    for path in row['feature_paths']:
        new_path = path
        if mode == 'append':
            new_path.append(entity_to_feature_row_dict['{}_{}'.format(ext_prefix, row[ext_col_name])])
        if mode == 'prepend':
            new_path.insert(0, entity_to_feature_row_dict['{}_{}'.format(ext_prefix, row[ext_col_name])])
        new_paths.append(new_path)
    return new_paths


def standard_df_to_tf_input(samples_df, kg_uid_to_feature_row_dict, ext_uid_to_feature_row_dict):
    assert 'kg_uid' in samples_df.columns.values
    assert 'ext_uid' in samples_df.columns.values
    assert 'label' in samples_df.columns.values

    df = samples_df.copy()
    df['kg_uid'] = df['kg_uid'].apply(lambda x: kg_uid_to_feature_row_dict[x])
    df['ext_uid'] = df['ext_uid'].apply(lambda x: ext_uid_to_feature_row_dict[x])

    features = {'graph_elts': df.values[:, 0].reshape(-1, 1), 'text_elts': df.values[:, 1].reshape(-1, 1)}
    labels = df['label'].values

    return features, labels


def get_feature_submatrix(X_full: np.array, feature_row_to_entity_dict: Dict[int, str], num_features: int,
                          prefix: str = None) -> np.array:
    """
    Returns a matrix X which is a subset of the rows of X_full. The positions of the i-th row has to be given by
    `feature_row_to_entity_dict[i]`.

    Parameters
    ----------
    X_full: np.array, shape=[n, dim]
        The full feature matrix.
    feature_row_to_entity_dict: Dict[int, str]
        A dictionary containing the position of the rows. E.g. feature_row_to_entity_dict[i] holds the row number of
        `X_full` of the i-th row.
    num_features: int
        The amount of features
    prefix: str
        Prefix of the dictionary indices.

    Returns
    -------
    X: np.array, shape=[num_features, dim]
    """
    X = np.zeros((num_features, X_full.shape[1]))

    for i in range(num_features):
        uid = feature_row_to_entity_dict[i]
        if prefix:
            uid = uid.replace('{}_'.format(prefix), '')
        X[i, :] = X_full[int(uid)]

    assert np.abs(X).sum() != 0

    return X


def create_feature_matrix(X_kg: np.array, X_ext: np.array, feature_row_to_entity_dict: Dict[int, str],
                          num_features) -> np.array:
    """
    Takes two matrices, each containing features for either the kg or the external features and reduces them to one
    feature matrix. The mapping is done using the `feature_row_to_entity_dict` dictionary.

    Parameters
    ----------
    X_kg: np.array, shape=[num_kg_features, dim]
        Each row contains feature vector of a kg element.
    X_ext: np.array, shape=[num_ext_features, dim]
        Each row contains feature vector of a external element.
    feature_row_to_entity_dict: Dict[int, str]
        A dictionary containing the position of the rows. E.g. feature_row_to_entity_dict[i] holds the row number of
        `X_full` of the i-th row.
    num_features:
        The amount of features.

    Returns
    -------
    X: np.array, shape=[num_features, dim]
        The created feature matrix.
    """

    assert X_kg.shape[1] == X_ext.shape[1]

    X = np.zeros((num_features, X_kg.shape[1]))

    for i in range(num_features):
        uid = feature_row_to_entity_dict[i]
        if uid.startswith('kg'):
            uid = int(uid.replace('kg_', ''))
            X[i] = X_kg[uid]
        elif uid.startswith('ext'):
            uid = int(uid.replace('ext_', ''))
            X[i] = X_ext[uid]
        else:
            assert True == False

    assert np.abs(X).sum() != 0

    return X
