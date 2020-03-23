from .test_cluster_det import test_cluster_det
from .test_cluster_seg import test_cluster_seg
from .train_cluster_det import train_cluster_det
from .train_cluster_seg import train_cluster_seg

__factory__ = {
    'test_det': test_cluster_det,
    'test_seg': test_cluster_seg,
    'train_det': train_cluster_det,
    'train_seg': train_cluster_seg,
}


def build_handler(phase, stage):
    key_handler = '{}_{}'.format(phase, stage)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
