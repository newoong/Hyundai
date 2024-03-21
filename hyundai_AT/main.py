import os
import argparse
import numpy as np
import yaml
from torch.backends import cudnn
from utils.utils import *

from solver import Solver


#def str2bool(v):
#    return v.lower() in ('true')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config['model_save_path'])):
        os.mkdir(config['model_save_path'])
    solver = Solver(config)

    if config['mode'] == 'train':
        solver.train()
    elif config['mode'] == 'test':
        if solver.calculate:
            accuracy, precision, recall, f_score=solver.test()
        else:
            test_energy, rec_loss_list, indexes, output_list = solver.test()
            npy_name=solver.test_path.split('/')[-1][:-8]
            np.save(f'{solver.model_save_path}/{npy_name}_anom_score.npy',test_energy)
            np.save(f'{solver.model_save_path}/{npy_name}_RCE.npy',rec_loss_list)
            np.save(f'{solver.model_save_path}/{npy_name}_indexes.npy',indexes)
            np.save(f'{solver.model_save_path}/{npy_name}_output.npy',output_list)
            print('test_result_saved !')

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='train file')
    parser.add_argument('--config_file', type=str, required=True, help="config file path")
    args = parser.parse_args()

    with open(args.config_file, encoding = "utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
   

    print('------------ Options -------------')
    for k, v in sorted(config.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
