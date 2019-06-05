from .test_cluster_det import test_cluster_det


__factory__ = {
     'test_det': test_cluster_det,
}


def build_op(phase, stage):
    key_op = '{}_{}'.format(phase, stage)
    if key_op not in __factory__:
        raise KeyError("Unknown op:", key_op)
    return __factory__[key_op]
