from .dsgcn import dsgcn


__factory__ = {
     'dsgcn': dsgcn,
}


def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknown model:", name)
    return __factory__[name](*args, **kwargs)
