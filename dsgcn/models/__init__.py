from .dsgcn import dsgcn


__factory = {
     'dsgcn': dsgcn,
}


def build_model(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
