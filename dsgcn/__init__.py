from .test_cluster_det import test_cluster_det
from .train_cluster_det import train_cluster_det

__factory__ = {
    'test_det': test_cluster_det,
    'train_det': train_cluster_det,
}


def build_handler(phase, stage):
    key_handler = '{}_{}'.format(phase, stage)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
