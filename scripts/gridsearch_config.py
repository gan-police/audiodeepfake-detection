"""Return configuration for grid search."""


def get_config() -> dict:
    """Return config dictonary for grid search.

    Note: The keys must adhere to the args keys.
    """
    config = {
        "learning_rate": [0.0005],
        "weight_decay": [0.001],
        "wavelet": ["sym8"],
        "dropout_cnn": [0.7, 0.6],
        "dropout_lstm": [0.0, 0.2],
        "num_of_scales": [256, 512],
        "aug_contrast": [False, True],
    }

    return config
