# --------------------------------------------------------
# MG-SSAF
# Author: ShuaiYang
# Date: 20230322
# --------------------------------------------------------

from .MS_SSAF_B import MGSSAF


def build_model(cfg):

    model = MGSSAF()

    return model
