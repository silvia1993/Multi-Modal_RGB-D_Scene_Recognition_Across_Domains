import torch
import torchvision
import torch.nn.functional as F
from torch.optim import lr_scheduler

import random
from collections import OrderedDict

from model.encoder import Encoder
from model.decoder import Decoder

def get_scheduler(optimizer, cfg):
    
    decay_start = cfg.NITER
    decay_epochs = cfg.NITER_DECAY

    def lambda_rule(epoch):
        lr_l = 1 - max(0, epoch - decay_start - 1) / float(decay_epochs)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        
    return scheduler

def update_learning_rate(optimizers, schedulers, epoch=None):
    for scheduler in schedulers:
        scheduler.step(epoch)

    for optimizer in optimizers:
        print('default lr', optimizer.defaults['lr'])
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print('/////////learning rate = %.7f' % lr)

def get_current_errors(loss_meters, current=True):

    loss_dict = OrderedDict()

    for key, value in sorted(loss_meters.items(), reverse=True):

        if 'TEST' in key or 'VAL' in key or 'ACC' in key or value.val == 0 or 'LAYER' in key:
            continue
        if current:
            loss_dict[key] = value.val
        else:
            loss_dict[key] = value.avg

    return loss_dict

def print_current_errors(errors, epoch, i=None, t=None):
    if i is None:
        message = '(Training Loss_avg [Epoch:{0}]) '.format(epoch)
    else:
        message = '(epoch: {epoch}, iters: {iter}, time: {time:.3f}) '.format(epoch=epoch, iter=i, time=t)

    for k, v in errors.items():
        message += '{key}: {value:.3f} '.format(key=k, value=v)

    print(message)

def set_input(data, cfg, device):
    
    rgb_batch = data['A']
    depth_batch = data['B']
    img_names = data['img_name']
    rgb_batch = rgb_batch.to(device)
    depth_batch = depth_batch.to(device)

    AtoB = cfg.WHICH_DIRECTION == "AtoB"
    source_modal = rgb_batch if AtoB else depth_batch
    target_modal = depth_batch if AtoB else rgb_batch

    current_batch_size = rgb_batch.size(0)

    gt_labels = torch.LongTensor(data['label']).to(device)

    return source_modal, target_modal, current_batch_size, gt_labels

def get_input(data, cfg, device):
    
    rgb_batch = data['A']
    depth_batch = data['B']
    img_names = data['img_name']
    rgb_batch = rgb_batch.to(device)
    depth_batch = depth_batch.to(device)

    current_batch_size = rgb_batch.size(0)

    gt_labels = torch.LongTensor(data['label']).to(device)

    return rgb_batch, depth_batch, current_batch_size, gt_labels


def compute_content_loss(cfg, content_model, content_layers, generated_images, target_modal, criterion_content, weight):
    
    assert (generated_images.size(-1) == cfg.FINE_SIZE)

    #print("Content Loss Weight : ",weight)

    source_features = content_model((generated_images + 1) / 2, layers=content_layers)
    target_features = content_model((target_modal + 1) / 2, layers=content_layers)
    
    len_layers = len(content_layers)

    loss_fns = [criterion_content] * len_layers
    alpha = [1] * len_layers

    layer_wise_losses = [alpha[i] * loss_fns[i](source_feature, target_features[i])
                            for i, source_feature in enumerate(source_features)] * weight

    content_loss = sum(layer_wise_losses)

    return content_loss


def loadWithoutModule(path, device):
    
    temp_chk = torch.load(path, map_location=device)
        
    return temp_chk

def createGenerators(cfg, device, chk_rgb, chk_depth):

    print("Usage of Fake Data")

    E_gen_rgb = Encoder(cfg)
    D_gen_rgb = Decoder(cfg, E_gen_rgb)
    E_gen_depth = Encoder(cfg)
    D_gen_depth = Decoder(cfg, E_gen_depth)

    E_gen_rgb.eval()
    D_gen_rgb.eval()
    E_gen_depth.eval()
    D_gen_depth.eval()

    E_gen_rgb.load_state_dict(loadWithoutModule(chk_rgb + "trecg_AtoB_70.pth", device))
    D_gen_rgb.load_state_dict(loadWithoutModule(chk_rgb + "trecg_AtoB_70_D.pth", device))
    E_gen_depth.load_state_dict(loadWithoutModule(chk_depth + "trecg_BtoA_70.pth", device))
    D_gen_depth.load_state_dict(loadWithoutModule(chk_depth + "trecg_BtoA_70_D.pth", device))

    print("Weights properly loaded into images generators")

    E_gen_rgb = E_gen_rgb.to(device)
    D_gen_rgb = D_gen_rgb.to(device)
    E_gen_depth = E_gen_depth.to(device)
    D_gen_depth = D_gen_depth.to(device)
    
    return E_gen_rgb, D_gen_rgb, E_gen_depth, D_gen_depth


def imageGeneration(E_gen_rgb, D_gen_rgb, E_gen_depth, D_gen_depth, input_rgb, input_depth):

    with torch.no_grad():
        #rgb images
        rgb_encoded_features = E_gen_rgb(input_rgb)
        depth_generated_images = D_gen_rgb(rgb_encoded_features)
        #depth images
        depth_encoded_features = E_gen_depth(input_depth)
        rgb_generated_images = D_gen_depth(depth_encoded_features)

    input_num = len(depth_generated_images)
    indexes = [i for i in range(input_num)]
    rgb_random_index = random.sample(indexes, int(input_num * 0.3))
    depth_random_index = random.sample(indexes, int(input_num * 0.3))

    for i in rgb_random_index:
        input_rgb[i, :] = rgb_generated_images.data[i, :]
    for j in depth_random_index:
        input_depth[j, :] = depth_generated_images.data[j, :]

    return input_rgb, input_depth