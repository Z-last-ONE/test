# coding: utf-8


"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import random
import numpy as np
import torch

from tools.generate_data import generate_data
from utils.quick_start import quick_start
from utils.utils import init_seed

os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    task = 'Baby'  # Baby  Home_and_Kitchen  Electronics
    use_weighting = True
    use_uu_ii_edge = True
    use_image_to_text_feature = True
    user_feature = False
    init_seed(999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MODEL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default=task, help='name of datasets')

    config_dict = {
        # 'dropout': [0.2],
        # 'reg_weight': [1e-04, 1e-03],
        'learning_rate': [0.0001],
        'reg_weight': [0.01],
        # 'n_layers': [2],
        # 'reg_weight': [0.01],
        'gpu_id': 0,
        'use_weighting': use_weighting,
        'user_feature': user_feature,
        'use_uu_ii_edge': use_uu_ii_edge,
        'use_image_to_text_feature': use_image_to_text_feature
    }
    args, _ = parser.parse_known_args()
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
