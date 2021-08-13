import warnings

# base configuration, might be considered as the abstract class
class DefaultConfig:
    # GPU / CPU
    GPU_IDS = None  # slipt different gpus with comma
    nTHREADS = 8
    WORKERS = 8

    # MODEL
    MODEL = 'trecg'
    ARCH = 'vgg11_bn'
    PRETRAINED = 'imagenet'
    CONTENT_PRETRAINED = 'imagenet'
    NO_UPSAMPLE = False  # set True when evaluating baseline
    FIX_GRAD = False

    # PATH
    DATA_DIR_TRAIN = '/datasets/new_sun_rd/kv2' #case raw_depth

    DATA_DIR_TRAIN_2 = None # case no KXtoR

    DATA_DIR_VAL = '/datasets/new_sun_rd/xtion' # case raw_depth

    CHECKPOINTS_DIR = './checkpoints'
    LOG_PATH = None

    # DATA
    WHICH_DIRECTION = None
    NUM_CLASSES = 19
    BATCH_SIZE = 48
    LOAD_SIZE = 256
    FINE_SIZE = 224
    FLIP = True
    FAKE_DATA_RATE = 0.3

    # OPTIMIZATION
    LR = 2e-4

    # TRAINING / TEST
    RESUME_PATH_A = None
    RESUME_PATH_B = None
    START_EPOCH = 1
    ROUND = 1
    NITER = 10
    NITER_DECAY = 40
    NITER_TOTAL = 50
    USE_FAKE_DATA = False

    # classfication task
    ALPHA_CLS = 1

    # translation task
    WHICH_CONTENT_NET = 'vgg11_bn'
    CONTENT_LAYERS = ['l0', 'l1', 'l2']
    ALPHA_CONTENT = 10

    # Across Domain task
    FILTER_BEDROOM = False
    MIXED_SOURCE = False
    ALPHA_TARGET_CONTENT = 3

    def parse(self, kwargs, printConfig = True):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut {0}".format(k))
            setattr(self, k, v)

        if printConfig:
            print('user config:')
            for k, v in self.__class__.__dict__.items():
                if not k.startswith('__'):
                    print(k, ':', getattr(self, k))
