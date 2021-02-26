import os


def get_model_path():
    """
    Simple retrieval function.
    Returns MODEL_FILE or raises OSError.
    """
    model_path = os.getenv("TEST_MODEL_FILE", default=None)
    if model_path is None:
        raise OSError("TEST_MODEL_FILE environment is not set.")
    return model_path
