import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, label

from .inference_model import InferenceModel
from med_lightning.datasets.transforms import SplitCombiner


np.set_printoptions(suppress=True)


class NoduleDetectionInference(InferenceModel):
    def __init__(self, config_path, pretrain_weights, gpu_device):
        super().__init__(config_path, gpu_device)

        # Arguments for preprocessing
        self.margin = 10
        self.resolution = (1, 1, 1)
        self.pad_value = 170

        # threshold is recorded in models/nodule_detection_yi_s3_b210/cpm_threshold_results_subset0.txt
        # CPM @ 4 FP/scan : 0.926097542
        # CPM @ 2 FP/scan : 0.980911999

        # TODO: Load this value from text file
        self.threshold = 0.980911999

        self.transforms = self._get_transforms(self.cfg['datasets']['lungnodule']['heavy_val']['transforms'])
        self.SplitCombiner = SplitCombiner(**self.cfg['datasets']['lungnodule']['heavy_val']['split_combine'])
        self.load_checkpoint(pretrain_weights)

    def forward(self, x):
        return self.networks.detection(x)

    def _inference(self, vol, mask, spacing, *args, **kwargs):
        vol_resampled, mask_resampled, extendbox = self.preprocess(vol, mask, spacing)
        data_output = {'data': vol_resampled}
        data_output = self.transforms(**data_output)

        # Add one dimension as batch_size to align how model get data from data_loader when training
        data_output['data'] = data_output['data'][np.newaxis]
        net_out = self.heavy_val_forward(batch=data_output)

        data = {
            'vol_resampled': vol_resampled,
            'mask_resampled': mask_resampled,
            'extendbox': extendbox
        }
        pred = self.post_process(net_out)

        return pred, data

    def process_output_to_world_coord(self, output, origin, spacing, extendbox, resolution):
        output = output.transpose(1, 0)
        d = output.device
        output[1:4] = output[1:4] + torch.from_numpy(np.expand_dims(extendbox[0, :, 0], 1)).to(d)
        output[1:4] = output[1:4] * torch.from_numpy(resolution.T).to(d) + torch.from_numpy(origin.T).to(d)
        output[4] = output[4] * resolution[0, 0]
        output = output.transpose(0, 1)

        # turn (confidence, z, y, x, diameter) into (confidence, x, y, z, diameter)
        output = output[:, [0, 3, 2, 1, 4]]

        return output

    def label_resampling(self, bboxes, spacing, extendbox, original_spacing=(1, 1, 1)):
        coord = bboxes[:, 0:3]
        diameter = bboxes[:, 3:]

        new_coord = (coord + extendbox[:, :1].T)
        new_coord = new_coord / spacing * original_spacing
        new_diameter = diameter / spacing[1] * original_spacing[1]
        new_bboxes = np.round(np.concatenate([new_coord, new_diameter], axis=1)).astype(int)

        return new_bboxes

    def preprocess(self, image, mask, spacing):
        # generate crop box coordinate
        # mask = self.post_process(mask)  # Noted that this had been done in lung_segment.py:post_processing

        newshape = np.round(np.array(mask.shape) * spacing / self.resolution).astype('int')
        xx, yy, zz = np.where(mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(self.resolution, 1)
        box = np.floor(box).astype('int')
        extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - self.margin], 0),
                               np.min([newshape, box[:, 1] + 2 * self.margin], axis=0).T]).T

        image = self.clip_and_normalize(image)

        image = image * (mask > 0)
        image[mask == 0] = self.pad_value
        image[image >= 210] = self.pad_value

        # volume/seg need resampling and slicing
        volume, _ = self.resample(image, spacing, self.resolution, order=1)
        mask, _ = self.resample(mask, spacing, self.resolution, order=0)

        volume = volume[extendbox[0, 0]:extendbox[0, 1],
                        extendbox[1, 0]:extendbox[1, 1],
                        extendbox[2, 0]:extendbox[2, 1]]
        mask = mask[extendbox[0, 0]:extendbox[0, 1],
                    extendbox[1, 0]:extendbox[1, 1],
                    extendbox[2, 0]:extendbox[2, 1]]

        image = volume[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        return image, mask, extendbox

    def clip_and_normalize(self, img, lung_window=(-1200, 600)):

        lungwin = np.array(lung_window)
        newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        newimg = (newimg * 255).astype('uint8')
        return newimg

    def resample(self, imgs, spacing, new_spacing, order=1):

        if len(imgs.shape) == 3:
            new_shape = np.round(imgs.shape * spacing / new_spacing)
            true_spacing = spacing * imgs.shape / new_shape
            resize_factor = new_shape / imgs.shape
            imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
            return imgs, true_spacing
        elif len(imgs.shape) == 4:
            n = imgs.shape[-1]
            newimg = []
            for i in range(n):
                slice = imgs[:, :, :, i]
                newslice, true_spacing = self.resample(slice, spacing, new_spacing)
                newimg.append(newslice)
            newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
            return newimg, true_spacing
        else:
            raise ValueError('wrong shape')

    def post_process(ret):
        """
            Three steps, include:
            1. Parse bbox from output of network.forward
            2. Re-order prediction by z-axis coordinate
            3. Filter by threshold
        """

        pred = np.array([
            bbox.cpu().numpy()
            for bbox in ret['bbox']]
        ).squeeze() # [50, 5], [score, z, y, x, d]

        pred = pred[np.argsort(pred[:, 1])]
        pred = pred[pred[:, 0] > threshold]

        return pred


if __name__ == '__main__':

    config_path = '../../models/nodule_detection_yi_s3_b210/model.yaml'
    pretrain_weights = ['../../models/nodule_detection_yi_s3_b210/weight.ckpt']
    detector = NoduleDetectionInference(config_path, pretrain_weights, 0)

    # Randon Input
    vol = np.ones((300, 300, 300))
    mask = np.ones((300, 300, 300))
    spacing = np.array((1, 1, 1))
    origin = np.array((0, 0, 0))

    ret, data = detector.inference(vol, mask, spacing)

    # World coord
    bbox = [bbox.cpu() for bbox in ret['bbox']]     # [50, 5]
    origin = origin[np.newaxis]                     # [1, 3]
    extendbox = data['extendbox'][np.newaxis]       # [1, 3, 2]
    resolution = np.array([[1, 1, 1]])              # [1, 3]

    world_bbox = detector.process_output_to_world_coord(bbox[0], origin, spacing, extendbox, resolution)
    print(world_bbox.data.numpy())
