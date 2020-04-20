from .test_gcn_v import test_gcn_v
from .test_gcn_e import test_gcn_e
from .train_gcn_v import train_gcn_v
from .train_gcn_e import train_gcn_e

__factory__ = {
    'test_gcn_v': test_gcn_v,
    'test_gcn_e': test_gcn_e,
    'train_gcn_v': train_gcn_v,
    'train_gcn_e': train_gcn_e,
}


def build_handler(phase, model):
    key_handler = '{}_{}'.format(phase, model)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
