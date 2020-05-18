import numpy as np
import torch.nn as nn
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, label

from .inference_model import InferenceModel
from med_lightning.datasets.transforms import SplitCombiner


class LungSegmentationInference(InferenceModel):
    def __init__(self, config_path, pretrain_weights, gpu_device):
        super().__init__(config_path, gpu_device)
        self.SplitCombiner = SplitCombiner(**self.cfg['datasets']['lung']['heavy_val']['split_combine'])
        self.transforms = self._get_transforms(self.cfg['datasets']['lung']['heavy_val']['transforms'])
        self.load_checkpoint(pretrain_weights)

        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

    def forward(self, x):
        net_out = self.networks.segmentation(x['data'])
        return {'seg': net_out}

    def _inference(self, vol, threshold=0.5, *args, **kwargs):
        data_output = {'data': vol}
        data_output = self.transforms(**data_output)

        data_output['data'] = data_output['data'].unsqueeze(0)
        net_out = self.heavy_val_forward(data_output)
        net_out['seg'] = self.sigmoid(self.upsample(net_out['seg']))
        net_out['mask'] = self.post_processing(net_out['seg'].detach().cpu().numpy(), threshold).astype(int)

        mask = net_out['mask'].squeeze()
        return mask

    def post_processing(self, mask, threshold):
        selem = np.ones((3, 3, 3))
        binary_img = binary_dilation(mask[0, 0] > threshold, selem=selem)

        label_image = label(binary_img, connectivity=2)
        region_area = np.array([(region.label, region.area) for region in regionprops(label_image)])
        sorted_region_area = region_area[np.argsort(-region_area[:, 1])]

        if len(sorted_region_area) > 1 and sorted_region_area[0, 1] < sorted_region_area[1, 1] * 4:
            label_image = np.logical_or(
                label_image == sorted_region_area[0, 0], label_image == sorted_region_area[1, 0])
        else:
            label_image = label_image == sorted_region_area[0, 0]

        return label_image


if __name__ == '__main__':
    config_path = '/host/med-lightning-deploy/models/lung_segmentation/model_revised.yaml'
    pretrain_weights = ['/host/med-lightning-deploy/models/lung_segmentation/weight.ckpt']
    segmentor = LungSegmentationInference(config_path, pretrain_weights, 1)
    ret = segmentor.inference(np.ones([1, 33, 512, 512]), 0)
    print(ret['seg'].shape)
