import numpy as np
import torch
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import regionprops
from skimage.morphology import label, binary_closing, disk\

from .nodule_detect import NoduleDetectionInference

np.set_printoptions(suppress=True)


class NoduleDetectionBodyInference(NoduleDetectionInference):
    def _inference(self, vol, mask, spacing, *args, **kwargs):
        net_out, data = super()._inference(vol, mask, spacing, *args, **kwargs)
        # vol_resampled, mask_resampled, extendbox = self.preprocess(vol, mask, spacing)
        # data_output = {'data': vol_resampled}
        # data_output = self.transforms(**data_output)

        # # Add one dimension as batch_size to align how model get data from data_loader when training
        # data_output['data'] = data_output['data'][np.newaxis]
        # net_out = self.heavy_val_forward(batch=data_output)

        # data = {
        #     'vol_resampled': vol_resampled,
        #     'mask_resampled': mask_resampled,
        #     'extendbox': extendbox
        # }

        net_out = self.filter_outside_mask(net_out, mask)

        return net_out, data

    def filter_outside_mask(self, net_out, mask):
        # TODO: filter prediction which is outside mask.
        return net_out

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

    def preprocess(self, image, mask, spacing):
        # generate crop box coordinate
        body_label, percentages = self.extract_whole_body(image)
        body_mask = body_label == 1
        body_masked_image = np.where(body_mask, image, 0)

        # Window (-1200 ~ 600) and bone removal (value > 300)
        cliped_image = np.clip(body_masked_image, -1200, 600)
        bone_removed = np.where(cliped_image < 300, cliped_image, 0)

        # Normalize
        normailzed = (bone_removed - bone_removed.min()) / (bone_removed.max() - bone_removed.min())
        normailzed = (normailzed * 255).round().astype('uint8')

        extendbox = self.get_extendbox(body_mask, spacing)
        image = normailzed

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

    def get_extendbox(self, mask, spacing):
        newshape = np.round(np.array(mask.shape) * spacing / self.resolution).astype('int')
        xx, yy, zz = np.where(mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(self.resolution, 1)
        box = np.floor(box).astype('int')
        extendbox = np.vstack([
            np.max([[0, 0, 0], box[:, 0] - self.margin], 0),
            np.min([newshape, box[:, 1] + 2 * self.margin], axis=0).T
        ]).T

        return extendbox

    def extract_whole_body(self, image):
        body_mask = np.zeros_like(image)
        percentages = []

        for idx, slic in enumerate(image):
            thresholded = slic > -400

            # Only do this when there is any area is greater than -400
            if thresholded.any():
                label_image = label(thresholded)
                areas = np.array([(r.area, r.label) for r in regionprops(label_image)])
                arg_area = np.argsort(areas[:, 0])

                label_largest = label_image == areas[arg_area][-1][1]
                label_largest = np.pad(label_largest, ((5, 5), (5, 5)), 'constant', constant_values=(0, 0))
                closing_label = binary_closing(label_largest, selem=disk(3))

                filled = binary_fill_holes(closing_label)
                body_mask[idx] = filled[5:-5, 5:-5]

                # opening_label = binary_opening(label_largest, selem=disk(5))
                # filled = binary_fill_holes(opening_label)

                # hull = convex_hull_image(filled)
                # percentages.append(np.sum(np.bitwise_xor(hull, filled))/np.sum(hull))
                percentages.append(0)

        return body_mask, np.array(percentages)


if __name__ == '__main__':

    config_path = '../../models/nodule_detection_fbody_closing/model.yaml'
    pretrain_weights = ['../../models/nodule_detection_fbody_closing/weight.ckpt']

    detector = NoduleDetectionBodyInference(config_path, pretrain_weights, 0)

    # Load JB Data
    # vol = np.load('/host/JB/JBsample/DCF1131050515_202002270006.npy')
    # spacing = np.load('/host/JB/JBsample/DCF1131050515_202002270006_spacing.npy')
    # with h5py.File('/host/JB/JBsample/mask.hdf5', 'r') as f:
    #     mask = np.array(f['DCF1131050515_202002270006']).squeeze()

    # LUNA Input
    # vol, origin, spacing, _ = load_itk_image('/host/lung_nodule/LUNA/raw/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.295420274214095686326263147663.mhd')
    # mask, _, _, _ = load_itk_image('/host/lung_nodule/LUNA/seg/1.3.6.1.4.1.14519.5.2.1.6279.6001.295420274214095686326263147663.mhd')

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
