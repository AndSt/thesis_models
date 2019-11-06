import tensorflow as tf


def checkpoint_config(path: str = 'data/models', gpu: int = 1):
    """Generate RunConfig to run estimator model.

    Parameters
    ----------
    path : str
        Path where the models are stored. Currently disabled
    gpu : int
        GPU ID

    Returns
    -------
    tf.estimator.RunConfig
        RunConfig for our models.

    """

    session_config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True)
    strategy = tf.contrib.distribute.OneDeviceStrategy(device='/device:GPU:{}'
                                                       .format(gpu))
    config = tf.estimator.RunConfig(
        # model_dir='{}'.format(path),
        # tf_random_seed=None,
        save_summary_steps=1000,
        save_checkpoints_steps=1500,
        # save_checkpoints_secs=None,
        session_config=session_config,
        keep_checkpoint_max=0,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=1000,
        train_distribute=strategy,
        # device_fn=None,
        # protocol=None,
        # eval_distribute=None,
        # experimental_distribute=None,
        # experimental_max_worker_delay_secs=None
    )
    return config
