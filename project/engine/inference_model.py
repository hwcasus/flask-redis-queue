import timeit
from collections import OrderedDict

import torch
import yaml

from med_lightning.builder import builder
from med_lightning.datasets import transforms


def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = timeit.default_timer()
        retval = func(*args, **kwargs)
        end_ts = timeit.default_timer()
        elapsed_time = (end_ts - beg_ts)
        return retval, elapsed_time
    return wrapper


class InferenceModel():
    def __init__(self, config_path, gpu_device):
        with open(config_path, 'r') as f:
            self.cfg = yaml.load(f, yaml.Loader)

        self.device = torch.device(f'cuda:{gpu_device}')
        self.networks = builder.build_networks(self.cfg['model']['networks']).to(self.device)

    @time_usage
    def inference(self, *args, **kwargs):
        return self._inference(*args, **kwargs)

    def _inference(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Please inherit this class and implement it")

    def load_checkpoint(self, checkpoint_paths):
        """
        Copied from med_lightning/models/basemodel.py:L77 new version
        Load checkpoints if existed

        """
        model_dict = self.networks.state_dict()
        for ckpt_path in checkpoint_paths:
            print(f'load checkpoint from {ckpt_path}')

            ckpt_dict = dict()
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict']

            for key, value in ckpt.items():
                if key in model_dict.keys():
                    ckpt_dict[key] = value
                elif key.split('.', 1)[-1] in model_dict.keys():
                    ckpt_dict[key.split('.', 1)[-1]] = value
                else:
                    print(f'Weight {key} not loaded')

            model_dict.update(ckpt_dict)

        self.networks.load_state_dict(model_dict)

    def dep_load_checkpoint(self, checkpoint_paths):
        """
        Copied from med_lightning/models/basemodel.py:L77
        Load checkpoints if existed

        """
        checkpoint = OrderedDict()
        for network_name in checkpoint_paths:
            checkpoint.update(torch.load(network_name, map_location=lambda storage, loc: storage)['state_dict'])

        loaded_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            if key in self.networks.state_dict().keys():
                loaded_checkpoint[key] = value
            # This is workaround, The way i build model will cause name inconsistency
            elif key.split('.', 1)[-1] in self.networks.state_dict().keys():
                loaded_checkpoint[key.split('.', 1)[-1]] = value
            else:
                print("Weight {} not loaded".format(key))
        self.networks.load_state_dict(loaded_checkpoint)

    def heavy_val_forward(self, batch, split_batch_size=1):
        """
        Copied from med_lightning/models/basemodel.py:L190
        Heavy validation forward is a progress to simulate
        a whole volume as input, not just a single cropped volume.
        Args:
            batch: [dict]
            split_batch_size: [int]
        Returns:
            net_out: [dict]
        """

        # Split a whole batch into seperate mini batches
        split_batches = self.SplitCombiner.split(batch)

        # Convert batch to split_batch
        split_batches = self._split_batch(split_batches, self.SplitCombiner.split_batch_size)

        split_net_outs = []
        with torch.no_grad():
            for split_batch in split_batches:
                split_batch['data'] = split_batch['data'].to(self.device)
                split_net_out = self.forward(split_batch)
                split_net_outs.append(split_net_out)                            # split_net_outs: [list] (len(split_batches), dict)

        # concat each split
        net_out = self._concat_split_net_outs(split_net_outs)               # net_out: [dict] (key: [torch.Tensor] (len(split_batchs, ...)))

        # remove split net out and clear gpu memory
        del split_net_outs
        torch.cuda.empty_cache()

        # Combine mini net_out into a whole net_out
        net_out = self.SplitCombiner.combine(net_out)

        # Recovery target key element with original one
        target_key = self.SplitCombiner.target_key
        batch[target_key] = batch.pop(f'origin_{target_key}')

        return net_out

    def _split_batch(self, batch, split_batch_size=1):
        """
        Copied from med_lightning/models/basemodel.py:L228
        This method is mainly for heavy_forward to split incoming batch into pieces

        Args:
            batch: [dict], the batch provided by dataloader
        Returns:
            split_batches: [list], the list of split batch (each one is a dictionary)
        """
        target_key = self.SplitCombiner.target_key

        # get number of splits and number of split batches to create
        n_splits = batch[target_key].size(0)

        n_split_batches = n_splits // split_batch_size \
                            if (n_splits % split_batch_size) == 0 \
                            else (n_splits // split_batch_size) + 1

        # batch to split batches
        split_batches = []

        for i_split in range(n_split_batches):

            # create an empty split
            single_split_batch = OrderedDict()

            # assign values to each split
            for key, value in batch.items():

                if key == target_key:
                    single_split_batch[key] = \
                        value[split_batch_size * i_split:split_batch_size * (i_split + 1)]
                else:
                    single_split_batch[key] = value

            # add to split batches
            split_batches.append(single_split_batch)

        return split_batches

    def _concat_split_net_outs(self, split_net_outs):
        """
        Copied from med_lightning/models/basemodel.py:L268
        This method is mainly for heavy_forward
        to concatenate split_net_output back into a whole model output
        Args:
            split_net_outs: []
        Returns:
            net_out: [dict]
                The keys in net_out are defined by your network output
        """

        net_out = OrderedDict()

        # get the keys in each split net out
        net_out_keys = split_net_outs[0].keys()

        for key in net_out_keys:
            if isinstance(split_net_outs[0][key], torch.Tensor):
                # all tensors corresponding to this key should be in the same shape
                net_out[key] = torch.cat([split_out[key]
                                         for split_out in split_net_outs], dim=0).detach().cpu()   # net_out[key]: [torch.Tensor] (splits_num, ...)

            elif isinstance(split_net_outs[0][key], list):
                # each element from individual list will be gathered into one list

                extended_list = []
                for split_out in split_net_outs:
                    extended_list.extend(split_out[key])
                # add batch dimension back
                net_out[key] = extended_list                                                       # net_out[key]: [list] (splits_num, torch.Tensor)

            else:
                # other types
                appended_list = []
                for split_out in split_net_outs:
                    appended_list.append(split_out[key])
                # add batch dimension back
                net_out[key] = [appended_list]

        return net_out

    def _get_transforms(self, transforms_list):
        """Copied from med_lightning/datasets/datasets.py:L142"""
        if transforms_list is None:
            return None

        transforms_methods = []
        for method in transforms_list:
            transforms_methods.append(eval(f'transforms.{method[0]}(**method[1])'))
        transforms_methods = transforms.Compose(transforms_methods)

        return transforms_methods


def clean_training_configs(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, yaml.Loader)

    # Clean redundant part of yaml
    cfg.pop('exp')
    cfg['datasets']['global']['modes'] = ['heavy_val']
    dataset_key = [key for key in cfg['datasets'].keys() if key != 'global']
    for key in dataset_key:
        cfg['datasets'][key].pop('train', None)
        cfg['datasets'][key].pop('val', None)

    cfg['model'].pop('criterions', None)
    cfg['model'].pop('metrics', None)
    cfg['model'].pop('optimizers', None)

    return cfg
