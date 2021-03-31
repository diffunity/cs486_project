import os

import torch
import numpy as np

from gym.utils import seeding
from stable_baselines3.common.utils import set_random_seed


def set_seed(seed=seed):
    os.environ['PYTHONHASHSEED']=str(seed) 

    _,seed = seeding.np_random(seed)
    random.seed(seed)
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
