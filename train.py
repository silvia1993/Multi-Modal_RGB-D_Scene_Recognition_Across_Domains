import os
import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

from collections import Counter
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from config.default_config import DefaultConfig
from config.resnet_sunrgbd_config import RESNET_SUNRGBD_CONFIG

import data.aligned_conc_dataset as dataset
from data import DataProvider

from model.encoder import Encoder
from model.decoder import Decoder
from model.classifier import Classifier
from model import networks

import util.utils as util
from util.average_meter import AverageMeter
from util.utilities import get_scheduler, update_learning_rate, get_input, get_current_errors, print_current_errors, compute_content_loss, createGenerators, imageGeneration

#######
# Configs Reading
#######

cfg = DefaultConfig()

cfg.parse(RESNET_SUNRGBD_CONFIG().args())

args = {
    'chk_rgb': "path/to/rgb/checkpoint/folder/",
    'chk_depth': "path/to/depth/checkpoint/folder/",
}

print("chk_rgb : ", args["chk_rgb"])
print("chk_depth : ", args["chk_depth"])

#######
# Dataset and Dataloader Creation
#######

source_dataset = dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_TRAIN, transform=transforms.Compose([
    dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
    dataset.RandomCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
    dataset.RandomHorizontalFlip(),
    dataset.ToTensor(),
    dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

]))

target_dataset = dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_VAL, transform=transforms.Compose([
    dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
    dataset.CenterCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
    dataset.ToTensor(),
    dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

]))

source_loader = DataProvider(cfg, dataset=source_dataset)

target_loader = DataProvider(cfg, dataset=target_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

#######
# Further Initializations
#######

# tensorboard
writer = SummaryWriter(log_dir=cfg.LOG_PATH)  

save_dir = os.path.join(cfg.CHECKPOINTS_DIR, cfg.MODEL,
                            str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))))
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

util.mkdir(save_dir)

#######
# Network and Optimizers
#######

device = torch.device("cuda")
#device = torch.device("cpu")

E_rgb = Encoder(cfg)
D_rgb = Decoder(cfg, E_rgb)
E_depth = Encoder(cfg)
D_depth = Decoder(cfg, E_depth)
C = Classifier(cfg)

if cfg.USE_FAKE_DATA:
    E_gen_rgb, D_gen_rgb, E_gen_depth, D_gen_depth = createGenerators(cfg, device, chk_rgb=args["chk_rgb"], chk_depth=args["chk_depth"])

# Classification Loss
criterion_cls = torch.nn.CrossEntropyLoss()

# Semantic Content Loss
criterion_content = torch.nn.L1Loss()
content_model = networks.Content_Model(cfg, criterion_content).to(device)
assert(cfg.CONTENT_LAYERS)
content_layers = cfg.CONTENT_LAYERS.split(',')

opt_E_rgb = torch.optim.Adam(E_rgb.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
opt_D_rgb = torch.optim.Adam(D_rgb.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
opt_E_depth = torch.optim.Adam(E_depth.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
opt_D_depth = torch.optim.Adam(D_depth.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
opt_C = torch.optim.Adam(C.parameters(), lr=cfg.LR, betas=(0.5, 0.999))

optims_list = [opt_E_rgb, opt_D_rgb, opt_E_depth, opt_D_depth, opt_C]

# set_log_data

loss_meters = OrderedDict()
log_keys = [
    'TRAIN_SEMANTIC_LOSS_RtoD',
    'TRAIN_SEMANTIC_LOSS_DtoR',
    'TRAIN_CLS_ACC',
    'TRAIN_CLS_LOSS',
    'TRAIN_TARGET_SEMANTIC_LOSS_RtoD',
    'TRAIN_TARGET_SEMANTIC_LOSS_DtoR',
    'TRAIN_TOTAL_LOSS',
    'VAL_CLS_ACC',
    'VAL_CLS_MEAN_ACC',
    'VAL_CLS_TOT_ACC'
]

for item in log_keys:
    loss_meters[item] = AverageMeter()

# set_schedulers

schedulers = [get_scheduler(optimizer, cfg) for optimizer in optims_list]

# nn.DataParallel

E_rgb = nn.DataParallel(E_rgb).to(device)
D_rgb = nn.DataParallel(D_rgb).to(device)
E_depth = nn.DataParallel(E_depth).to(device)
D_depth = nn.DataParallel(D_depth).to(device)
C = nn.DataParallel(C).to(device)

net_list = [E_rgb, D_rgb, E_depth, D_depth, C]

best_prec = 0
best_epoch = 0
best_prec_tot = 0
best_epoch_tot = 0

for epoch in range(cfg.START_EPOCH, cfg.NITER_TOTAL + 1):

    imgs_all = []
    pred_index_all = []
    target_index_all = []

    start_time = time.time()
    target_iterator = iter(target_loader) 

    update_learning_rate(optims_list, schedulers, epoch=epoch)

    for net in net_list:
        net.train()

    for key in loss_meters:
        loss_meters[key].reset()
    
    iters = 0
    for i, data in enumerate(source_loader):

        #### set_input

        rgb_batch, depth_batch, current_batch_size, gt_labels = get_input(data, cfg, device)

        if cfg.USE_FAKE_DATA:

            rgb_batch, depth_batch = imageGeneration(E_gen_rgb, D_gen_rgb, E_gen_depth, D_gen_depth, 
                                                        rgb_batch, depth_batch)

        # other initialization

        iter_start_time = time.time()
        iters += 1

        for optim in optims_list:
            optim.zero_grad()

        # forward

        #rgb images
        rgb_encoded_features = E_rgb(rgb_batch)
        depth_generated_images = D_rgb(rgb_encoded_features)
        #depth images
        depth_encoded_features = E_depth(depth_batch)
        rgb_generated_images = D_depth(depth_encoded_features)

        concat = torch.cat((rgb_encoded_features, depth_encoded_features), 1).to(device)

        predicted_labels = C(concat)

        # classification loss

        loss_total = torch.zeros(1)
        loss_total = loss_total.to(device)

        cls_loss = criterion_cls(predicted_labels, gt_labels) * cfg.ALPHA_CLS

        loss_total = loss_total + cls_loss

        cls_loss = round(cls_loss.item(), 4)

        loss_meters['TRAIN_CLS_LOSS'].update(cls_loss, current_batch_size)

        prec1 = util.accuracy(predicted_labels.data, gt_labels, topk=(1,))
        
        loss_meters['TRAIN_CLS_ACC'].update(prec1[0].item(), current_batch_size)

        ## semantic loss

        # Rgb to Depth

        content_loss_rgb_to_depth = compute_content_loss(cfg, content_model, content_layers, depth_generated_images, depth_batch, criterion_content, cfg.ALPHA_CONTENT)

        loss_total = loss_total + content_loss_rgb_to_depth

        loss_meters['TRAIN_SEMANTIC_LOSS_RtoD'].update(content_loss_rgb_to_depth.item(), current_batch_size)
        
        # Depth to Rgb
        
        content_loss_depth_to_rgb = compute_content_loss(cfg, content_model, content_layers, rgb_generated_images, rgb_batch, criterion_content, cfg.ALPHA_CONTENT)

        loss_total = loss_total + content_loss_depth_to_rgb

        loss_meters['TRAIN_SEMANTIC_LOSS_DtoR'].update(content_loss_depth_to_rgb.item(), current_batch_size)

        # optimization

        loss_total.backward()

        #################################################

        #Operation on the target domain
        
        try: #try to extract from the iterator
            target_data = next(target_iterator)
        except StopIteration: #exception raised when the iterator terminate data. 
            target_iterator = iter(target_loader)  
            target_data = next(target_iterator)
        
        #### set_input on target_data

        rgb_batch, depth_batch, current_batch_size, gt_labels = get_input(target_data, cfg, device)
        
        if cfg.USE_FAKE_DATA:

            rgb_batch, depth_batch = imageGeneration(E_gen_rgb, D_gen_rgb, E_gen_depth, D_gen_depth, 
                                                        rgb_batch, depth_batch)

        # forward

        #rgb images
        rgb_encoded_features = E_rgb(rgb_batch)
        depth_generated_images = D_rgb(rgb_encoded_features)
        #depth images
        depth_encoded_features = E_depth(depth_batch)
        rgb_generated_images = D_depth(depth_encoded_features)

        ## target semantic loss

        # Rgb to Depth

        content_loss_rgb_to_depth = compute_content_loss(cfg, content_model, content_layers, depth_generated_images, depth_batch, criterion_content, cfg.ALPHA_TARGET_CONTENT)

        loss_total = loss_total + content_loss_rgb_to_depth

        loss_meters['TRAIN_TARGET_SEMANTIC_LOSS_RtoD'].update(content_loss_rgb_to_depth.item(), current_batch_size)
        
        content_loss_rgb_to_depth.backward()

        # Depth to Rgb

        content_loss_depth_to_rgb = compute_content_loss(cfg, content_model, content_layers, rgb_generated_images, rgb_batch, criterion_content, cfg.ALPHA_TARGET_CONTENT)

        loss_total = loss_total + content_loss_depth_to_rgb

        loss_meters['TRAIN_TARGET_SEMANTIC_LOSS_DtoR'].update(content_loss_depth_to_rgb.item(), current_batch_size)
        
        content_loss_depth_to_rgb.backward()

        #####################################################

        loss_meters['TRAIN_TOTAL_LOSS'].update(loss_total.item(), current_batch_size)

        for optim in optims_list:
            optim.step()
        
        if epoch % 5 == 0 and iters % 5 == 0:
            errors = get_current_errors(loss_meters)
            t = (time.time() - iter_start_time)
            print_current_errors(errors, epoch, i, t)

    
    # save model
    model_basename = '{0}_{1}_{2}'.format(cfg.MODEL, cfg.WHICH_DIRECTION, cfg.NITER_TOTAL)
    
    torch.save(E_rgb.module.state_dict(), os.path.join(save_dir, f'{model_basename}_E_rgb.pth'))
    torch.save(D_rgb.module.state_dict(), os.path.join(save_dir, f'{model_basename}_D_rgb.pth'))
    torch.save(E_depth.module.state_dict(), os.path.join(save_dir, f'{model_basename}_E_depth.pth'))
    torch.save(D_depth.module.state_dict(), os.path.join(save_dir, f'{model_basename}_D_depth.pth'))
    torch.save(C.module.state_dict(), os.path.join(save_dir, f'{model_basename}_C.pth'))

    print('iters in one epoch:', iters)

    train_errors = get_current_errors(loss_meters, current=False)
    print('#' * 10)
    print_current_errors(train_errors, epoch)
    
    # write loss

    writer.add_scalar('LR', opt_E_rgb.param_groups[0]['lr'], global_step=epoch)
            
    writer.add_scalar('TRAIN_TOTAL_LOSS', loss_meters['TRAIN_TOTAL_LOSS'].avg, 
                        global_step=epoch)

    writer.add_scalar('TRAIN_CLS_LOSS', loss_meters['TRAIN_CLS_LOSS'].avg,
                        global_step=epoch)
   
    writer.add_scalar('TRAIN_CLS_ACC', loss_meters['TRAIN_CLS_ACC'].avg,
                        global_step=epoch)

    writer.add_scalar('TRAIN_SEMANTIC_LOSS_RtoD', loss_meters['TRAIN_SEMANTIC_LOSS_RtoD'].avg,
                        global_step=epoch)
    
    writer.add_scalar('TRAIN_SEMANTIC_LOSS_DtoR', loss_meters['TRAIN_SEMANTIC_LOSS_DtoR'].avg,
                        global_step=epoch)

    writer.add_scalar('TRAIN_TARGET_SEMANTIC_LOSS_RtoD', loss_meters['TRAIN_TARGET_SEMANTIC_LOSS_RtoD'].avg,
                                       global_step=epoch)

    writer.add_scalar('TRAIN_TARGET_SEMANTIC_LOSS_DtoR', loss_meters['TRAIN_TARGET_SEMANTIC_LOSS_DtoR'].avg,
                                       global_step=epoch)

    print('Training Time: {0} sec'.format(time.time() - start_time))
    
    # switch to evaluate mode
    for net in net_list:
        net.eval()

    imgs_all = []
    pred_index_all = []
    target_index_all = []

    with torch.no_grad():

        print('# Cls val images num = {0}'.format(len(target_loader.dataset.imgs)))
        
        for i, data in enumerate(target_loader):
            
            rgb_batch, depth_batch, current_batch_size, gt_labels = get_input(data, cfg, device)

            #rgb images
            rgb_encoded_features = E_rgb(rgb_batch)
            #depth images
            depth_encoded_features = E_depth(depth_batch)

            concat = torch.cat((rgb_encoded_features, depth_encoded_features), 1).to(device)

            predicted_labels = C(concat)

            pred, pred_index = util.process_output(predicted_labels.data)

            pred_index_all.extend(list(pred_index))

            target_index_all.extend(list(gt_labels.cpu().numpy()))

            # accuracy
            prec1 = util.accuracy(predicted_labels.data, gt_labels, topk=(1,))

            loss_meters['VAL_CLS_ACC'].update(prec1[0].item(), current_batch_size)


    # Mean ACC
    mean_acc = util.mean_acc(np.array(target_index_all), np.array(pred_index_all),
                            cfg.NUM_CLASSES,
                            target_loader.dataset.classes)
    print('mean_acc: [{0}]'.format(mean_acc))

    acc_tot = accuracy_score(pred_index_all,target_index_all) * 100
    print(f"total_acc: {acc_tot}")

    print('Mean Acc Epoch <{epoch}> * Prec@1 <{mean_acc:.3f}> '
            .format(epoch=epoch, mean_acc=mean_acc))
    
    loss_meters['VAL_CLS_MEAN_ACC'].update(mean_acc)
    loss_meters['VAL_CLS_TOT_ACC'].update(acc_tot)

    writer.add_scalar('VAL_CLS_ACC', loss_meters['VAL_CLS_ACC'].avg,
                            global_step=epoch)
    writer.add_scalar('VAL_CLS_MEAN_ACC', loss_meters['VAL_CLS_MEAN_ACC'].avg,
                            global_step=epoch)
    writer.add_scalar('VAL_CLS_TOT_ACC', loss_meters['VAL_CLS_TOT_ACC'].avg,
                            global_step=epoch)

    assert (len(pred_index_all) == len(target_loader))

    is_best = mean_acc > best_prec
    best_prec = max(mean_acc, best_prec)
    best_epoch = epoch if is_best else best_epoch

    is_best_tot = acc_tot > best_prec_tot
    best_prec_tot = max(acc_tot, best_prec_tot)
    best_epoch_tot = epoch if is_best_tot else best_epoch_tot

    print('End of Epoch {0} / {1} \t '
            'Time Taken: {2} sec'.format(epoch, cfg.NITER_TOTAL, time.time() - start_time))
    print('-' * 80)

print('best mean acc is {0} in epoch {1}'.format(best_prec, best_epoch))
print('best acc_tot is {0} in epoch {1}'.format(best_prec_tot, best_epoch_tot))

if writer is not None:
    writer.close()