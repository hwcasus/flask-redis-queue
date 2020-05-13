import numpy as np
import torch
import torch.nn as nn

from .inference_model import InferenceModel


class NoduleSegClassifyInference(InferenceModel):
    def __init__(self, config_path, pretrain_weights, gpu_device):
        super().__init__(config_path, gpu_device)
        self.batch_size = self.cfg['datasets']['global']['batch_size']
        self.gpu_device = gpu_device
        self.softmax = nn.Softmax(dim=1)

        transform_args = self.cfg['datasets']['lungnodule']['preprocess']['transforms']
        for idx, (name, args) in enumerate(transform_args):
            if name == 'Resample' and 'new_spacing' in args:
                self.new_spacing = np.array(args['new_spacing'])
        self.preprocess = self._get_transforms(transform_args)

        # This is to change random crop into centor crop by fixing the shift value range to 0
        transform_args = self.cfg['datasets']['lungnodule']['val']['transforms']
        for idx, (name, args) in enumerate(transform_args):
            if name == 'RandomCrop' and 'shift' in args:
                transform_args[idx][1]['shift'] = (0, 0, 0)
        self.transforms = self._get_transforms(transform_args)
        # self.load_checkpoint(pretrain_weights)

    def forward(self, x):
        _input = x['data'].to(self.gpu_device)
        net_out = self.networks.segmentation(_input)
        return net_out

    def _inference(self, vol, bboxes, spacing, batch_size=5, *args, **kwargs):
        raw_output = {'data': vol, 'spacing': np.asarray(spacing)}
        data_output = self.preprocess(**raw_output)
        data_list = []

        for bbox in bboxes:
            data_output_cp = data_output.copy()
            coord, diameter = bbox[:3], bbox[3]
            new_coord = coord / self.new_spacing
            new_diameter = [diameter / self.new_spacing[0]]
            data_output_cp['target_bbox'] = np.concatenate([new_coord, new_diameter])
            data_output_cp = self.transforms(**data_output_cp)

            data_output_cp.pop('target_bbox')
            data_output_cp.pop('spacing')
            data_output_cp['data'] = data_output_cp['data'].unsqueeze(0)
            data_list.append(data_output_cp.pop('data'))

        probs_list = []
        total_num = len(data_list) // batch_size + int(len(data_list) % batch_size > 0)
        for i in range(total_num):
            chunk = torch.cat(data_list[i * batch_size: (i+1) * batch_size], dim=0)
            data_output = {'data': chunk}
            net_out = self.forward(data_output)

            # mask = net_out['seg']
            scores = net_out['cls']
            probs = self.softmax(scores)
            probs_list.append(probs)

        probs = torch.cat(probs_list).detach()
        return probs


if __name__ == '__main__':
    config_path = '/host/med-lightning-deploy/models/nodule_segment_classification/model.yaml'
    pretrain_weights = [None]
    classifier = NoduleSegClassifyInference(config_path, pretrain_weights, 2)

    vol = np.zeros((1, 300, 300, 300))
    spacing = np.array([1., 1., 1.])
    bbox = np.array([
        [130, 130, 130, 10], [130, 130, 130, 3],
        [130, 130, 130, 10], [130, 130, 130, 3],
        [130, 130, 130, 10], [130, 130, 130, 3],
        [130, 130, 130, 10], [130, 130, 130, 3],
        [130, 130, 130, 10], [130, 130, 130, 3],
    ])
    ret = classifier.inference(vol, bbox, spacing)
    print(ret)
