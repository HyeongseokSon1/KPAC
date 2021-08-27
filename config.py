from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9


## train set location
config.TRAIN.folder_path = '/home/sonhs/data2/DeblurDataset/DefocusDeblur/train_dual_cropped7/' # need to change
config.VALID = edict()
config.VALID.folder_path = '/home/sonhs/data2/DeblurDataset/DefocusDeblur/test_dual_cropped/'  # need to change

## test set location
config.TEST = edict()
config.TEST.folder_path = '/home/sonhs/data2/DeblurDataset/DefocusDeblur/test_c/'  # need to change
# for dual
config.TEST.folder_path_c = '/home/sonhs/data2/DeblurDataset/DefocusDeblur/test_c/' # need to change
config.TEST.folder_path_l = '/home/sonhs/data2/DeblurDataset/DefocusDeblur/test_l/' # need to change
config.TEST.folder_path_r = '/home/sonhs/data2/DeblurDataset/DefocusDeblur/test_r/' # need to change

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
