import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__ == '__main__':
    from thop import profile, clever_format

    model = MODEL_REGISTRY.get("SGFANetModel")()
    input = torch.randn(1, 3, 720, 1280)  # 固定输入尺寸

    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")  # 例如：FLOPs: 2.469G（FLOPs = 2 * mult-adds）
    print(f"Params: {params}")