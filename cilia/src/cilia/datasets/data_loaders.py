# Data loader registry
_dataitem_loader_registry = {}

def register_data_loader(dataset_cls, name=None, algorithm=None):
    def decorator(func):
        key = (dataset_cls, name, algorithm)
        _dataitem_loader_registry[key] = func
        return func
    return decorator