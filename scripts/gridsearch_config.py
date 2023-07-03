"""Return configuration for grid search."""


def get_config() -> dict:
    """Return config dictonary for grid search.

    Note: The keys must adhere to the args keys.
    """
    config = {
        "learning_rate": [0.0001, 0.0005],
        "weight_decay": [0.0001, 0.001, 0.01],
        "epochs": [10],
    }

    return config
