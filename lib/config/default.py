from detectron2.config import get_cfg, CfgNode as CN

_C = get_cfg()

_C.AUTO_RESUME =  True

_C.GPUS = (0,)
_C.SEED = 24

_C.RECORD_BASE = ''
_C.OUTPUT_DIR = ''

_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AMP =  True
_C.PIN_MEMORY =  True
_C.TASK = "regression"

_C.K_FOLD = 10
_C.VAL_K_FOLD = 10
_C.RESUME_K = 0
_C.RESUME_INNER_K = 3

_C.METRIC_THRESHOLD = 0.7

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK =  False
_C.CUDNN.DETERMINISTIC = True
_C.CUDNN.ENABLED = True

_C.DATASET = CN()
_C.DATASET.DATASET = "CellByPatient"
_C.DATASET.DS_ROOT = ''
_C.DATASET.IMG_DIR = ''
_C.DATASET.ANNO_DIR = ''
_C.DATASET.TRAIN_ANNO_DIR = ''
_C.DATASET.VAL_ANNO_DIR = ''
_C.DATASET.TEST_ANNO_DIR = ''
_C.DATASET.MASK_DIR = ''
_C.DATASET.KFOLD_LIST = ''
_C.DATASET.IMG_EXT = 'png'
_C.DATASET.BALANCE = True
_C.DATASET.CELL_AUG =   False
_C.DATASET.MEAN = [0.5, 0.5, 0.5]
_C.DATASET.STD = [0.5, 0.5, 0.5]
_C.DATASET.SHUFFLE =  True

_C.MODEL = CN()
_C.MODEL.INIT_WEIGHTS =  True
_C.MODEL.MODEL = "resnet"
_C.MODEL.NUM_LAYERS = 18
_C.MODEL.PRETRAINED = ''
_C.MODEL.PRETRAINED_STRICT = False
_C.MODEL.IMG_SIZE = [128, 128]
_C.MODEL.IMG_SHAPE = [128, 128]
_C.MODEL.MASK_SHAPE = [150, 300]
_C.MODEL.NUM_MASKS = 1
_C.MODEL.POL_TO_BOX = False
_C.MODEL.TO_HSV =  False
_C.MODEL.GAUSS_BLUR = False
_C.MODEL.BILATERAL_FILTER = False
_C.MODEL.MINMAX =  False
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.ONE_HOT_ENC =  False
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "Adam"

_C.LOSS = CN()
_C.LOSS.LOSS = "mse"
_C.LOSS.REDUCTION = "mean"
_C.LOSS.ALPHA = 1.0
_C.LOSS.BETA = 1.0
_C.LOSS.GAMMA = 2.0
_C.LOSS.SMOOTH = 1e-6
_C.LOSS.DICE_RAT = 1.0

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 30
_C.TRAIN.OPTIMIZER = "adam"
_C.TRAIN.LR = 0.002
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [20, 25]
_C.TRAIN.WD = 0.0001
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV =  False

_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 16
_C.TEST.TEST_MODEL_FILE = ''

_C.DEBUG = CN()
_C.DEBUG.DEBUG =  True
_C.DEBUG.SAVE_BATCH_IMAGES_RESULTS =  True
_C.DEBUG.SAVE_EMBED_VIS =  True

_C.WANDB = CN()
_C.WANDB.PROJECT = "my-project"
_C.WANDB.NAME = "exp-hrnet-kfold"


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.ds_root:
        cfg.DS_ROOT = args.ds_root

    if args.record_base:
        cfg.RECORD_BASE = args.record_base

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    cfg.freeze()