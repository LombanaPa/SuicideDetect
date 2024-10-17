"""
Manage exception.
"""
class ModelException(Exception):
    """
    Return pass to get model
    """
    pass


class ModelNotLoadException(ModelException):
    """
    Return pass when don't find model
    """
    pass
