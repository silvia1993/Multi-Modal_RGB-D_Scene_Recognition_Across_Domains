import os

class RESNET_SUNRGBD_CONFIG:

    def args(self):

        ########### Quick Setup ############
        model = 'trecg'    # | fusion
        arch = 'resnet18'  # | resnet50
        content_model = 'resnet18'  # | resnet50
        pretrained = 'imagenet'
        content_pretrained = 'imagenet'

        gpus = '0'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size = 40

        log_path = 'summary'            # path for tensorboardX log file
        lr = 2e-4

        direction = 'AtoB'              # AtoB: RGB->Depth
        no_upsample = False             # True for removing Decoder network
        content_layers = '0,1,2,3,4'    # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 10              # coefficient for content loss
        fix_grad = False

        # use generated data while training
        use_fake = False

        # if we do fusion, we need two tregnets
        resume_path_A = None  # the path for RGB TrecgNet
        resume_path_B = None  # the path for Depth TrecgNet

        return {

            'GPU_IDS': gpus,
            'WHICH_DIRECTION': direction,
            'BATCH_SIZE': batch_size,
            'PRETRAINED': pretrained,

            'LOG_PATH': log_path,

            # MODEL
            'MODEL': model,
            'ARCH': arch,
            'NO_UPSAMPLE': no_upsample,
            'FIX_GRAD': fix_grad,

            # DATA
            'NUM_CLASSES': 10,
            'USE_FAKE_DATA': use_fake,

            # TRAINING / TEST
            'RESUME_PATH_A': resume_path_A,
            'RESUME_PATH_B': resume_path_B,
            'LR': lr,

            'NITER': 20,
            'NITER_DECAY': 50,
            'NITER_TOTAL': 70,

            # TRANSLATION TASK
            'WHICH_CONTENT_NET': content_model,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,

            # Across Domain task
            'FILTER_BEDROOM' : False,
            'MIXED_SOURCE' : False,
            'ALPHA_TARGET_CONTENT' : 3
        }
