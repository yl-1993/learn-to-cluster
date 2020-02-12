from .test_lgcn import test_lgcn
from .train_lgcn import train_lgcn

__factory__ = {
    'test_lgcn': test_lgcn,
    'train_lgcn': train_lgcn,
}


def build_handler(phase):
    key_handler = '{}_lgcn'.format(phase)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
