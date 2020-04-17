from .gcn_v_dataset import GCNVDataset
from .gcn_e_dataset import GCNEDataset

__factory__ = {
    'gcn_v': GCNVDataset,
    'gcn_e': GCNEDataset,
}


def build_dataset(model_type, cfg):
    if model_type not in __factory__:
        raise KeyError("Unknown dataset type:", model_type)
    return __factory__[model_type](cfg)
