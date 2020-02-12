from .lgcn import lgcn

__factory__ = {
    'lgcn': lgcn,
}


def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknown model:", name)
    return __factory__[name](*args, **kwargs)
