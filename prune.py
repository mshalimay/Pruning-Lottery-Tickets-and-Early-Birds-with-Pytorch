"""
Contains auxiliary functions for structured and unstructured pruning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn

from models import channel_selection

def apply_mask(mod, mask):
    if isinstance(mod, nn.Conv2d):
        mod.weight.data.mul_(mask)
    elif isinstance(mod, nn.Linear):
        mod.weight.data.mul_(mask)
    elif isinstance(mod, nn.BatchNorm2d):
        mod.weight.data.mul_(mask)
        mod.bias.data.mul_(mask)
    else:
        raise ValueError(f"Module of type {type(mod)} not recognized")

def unstruct_pruning(model, percent, masks, mods_to_prune=(nn.Conv2d)):
    # determine device 
    device = next(model.parameters()).device

    # collect all weights to be consider for threshold in pruning
    total_w = total_w_alive = 0
    abs_weights_ls= []
    for m in model.modules():
        if isinstance(m, mods_to_prune):
            # count total number of weights
            total_w += m.weight.data.numel()
            # count alive weights
            alive = m.weight.data.abs().clone().gt(0).float().to(device)
            total_w_alive += torch.sum(alive)
            # collect abs weights
            abs_weights_ls.append(m.weight.data.view(-1).abs())

    # concatenate all abs(weights) into a single tensor
    abs_weights = torch.cat(abs_weights_ls).to(device)
    thre = torch.quantile(abs_weights[abs_weights>0], percent)

    # # manual threshold calculation
    # sorted_w, _ = torch.sort(abs_weights)
    # thre_index = (total_w - total_w_alive) + int(total_w_alive * percent)
    # thre = sorted_w[int(thre_index)]

    # dictionary of masks (one per module)
    new_masks = {} if masks is None else masks

    # iterate over all network modules and prune
    pruned = 0
    zero_flag = False
    for k, (name, m) in enumerate(model.named_modules()):
        if not isinstance(m, mods_to_prune):
            continue
        # prune current module weights

        weight_copy = m.weight.data.abs().clone()
        # if previous masks exists, "accumulate" the masks (for example, iterative pruning)
        if masks is not None:
            # equivalent to np.where(abs(tensor) < percentile_value, 0, mask[step])
            # weights pruned in previous iteration remain pruned; new pruned are added to them
              # e.g.: if w_t > thre, but previously pruned, mask[w_t] = 0 (mask[w_t-1] * mask[w_t] = 0 * 1)
            mask = masks[name] * weight_copy.gt(thre).float().to(device)                 
        else:
            mask = weight_copy.gt(thre).float().to(device)

        # equivalente to: `pruned += torch.sum(mask == 0)`
        pruned += mask.numel() - torch.sum(mask)
        
        # prune current module weights and store mask
        apply_mask(m, mask)

        new_masks[name] = mask

        if mask.sum() == 0:
            zero_flag = True
        # print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
        #         format(k, mask.numel(), int(torch.sum(mask))))

    print(f'Pruned {pruned:.0f} / {total_w} ({100*pruned/total_w:.2f}%) of weights')
    return new_masks, zero_flag

# TODO: 
#   (i)  use the dictionary of masks, instead of original list implementation
#   (ii) check if CFG is correct for non-bottleneck case
#   (iii) check if `isinstance(m, nn.MaxPool2d)` is correct
def structured_pruning(model, percent, masks, mods_to_prune=(nn.BatchNorm2d)):
    masks, _ = unstruct_pruning(model, percent, masks, mods_to_prune=mods_to_prune)
    cfg = []
    cfg_masks = []
    # convert masks dictionary to a list to folow `network_slimming` implementation
    for (name, m) in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            n_channels = int(torch.sum(masks[name]))                        
            # if pruning has resulted in a layer with no channels, add one channel
            if n_channels == 0:
                cfg.append(1)
                new_mask = masks[name].clone()
                new_mask[0] = 1
                cfg_masks.append(new_mask)
            else:
                cfg.append(n_channels)
                cfg_masks.append(masks[name].clone())
    return cfg, cfg_masks


# TODO:
#   (i)  use the dictionary of masks, instead of original list implementation
#   (ii) alternative way to check if last convolution of residual block; see if correct for non-bottleneck case
def reinitialize_weights(model, masks, init_state_dict):
    # determine the device from the model's parameters
    device = next(model.parameters()).device

    # TODO: implement reinitialization for structured pruning to use in iterative pruning
    if not masks:
        return
    
    for name, m in model.named_modules():
        if name in masks:
            # ensure mask and model in same device
            mask = masks[name].to(device)  
            # reinit non-zero weights
            if mask.sum() > 0:  # if there are non-zero weights  
                non_zero_weights = init_state_dict[name + '.weight'].to(device) * mask
                m.weight.data.copy_(non_zero_weights)                

            # Reinitialize bias if present
            if m.bias is not None:
                initial_bias = init_state_dict[name + '.bias'].to(device)
                m.bias.data.copy_(initial_bias)


def reinit_network(new_model, old_model, cfg, cfg_mask):
    old_modules = list(old_model.modules())
    new_modules = list(new_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the 
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions. 
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

