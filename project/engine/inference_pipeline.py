from .lung_segment import LungSegmentationInference
from .nodule_classify import NoduleClassifyInference
from .nodule_detect import NoduleDetectionInference


class InferencePipeline(object):
    def __init__(self, top_k=5, version_code='0417'):
        # TODO: Make model init configurable (by yaml maybe)
        self.detector = NoduleDetectionInference(
            config_path='models/nodule_detection_yi_s3_b210/model.yaml',
            pretrain_weights=['models/nodule_detection_yi_s3_b210/weight.ckpt'],
            gpu_device=0
        )
        self.segmentor = LungSegmentationInference(
            config_path='models/lung_segmentation/model_revised.yaml',
            pretrain_weights=['models/lung_segmentation/weight.ckpt'],
            gpu_device=1
        )
        self.classifier = NoduleClassifyInference(
            config_path='models/nodule_classification/model_concat.yaml',
            pretrain_weights=['models/nodule_classification/weight.ckpt'],
            gpu_device=2
        )

        self.top_k = top_k
        self.version_code = version_code

    def inference(self, vol, spacing):
        # Segmentation
        mask, elapsed_time_seg = self.segmentor.inference(vol[np.newaxis])
        print(f'Lung Segmentation - Done with elapsed time = {elapsed_time_seg}')

        # Detection
        (pred, data), elapsed_time_det = self.detector.inference(vol, mask, spacing)
        print(f'Lung Nodule Detection - Done with elapsed time = {elapsed_time_det}')

        # Select Top K (Default: 10)
        scores = pred[:self.top_k, :1]
        bbox = pred[:self.top_k, 1:]

        # Label resampling, back to original spacing and do classification
        new_bbox = self.detector.label_resampling(bbox, spacing, extendbox=data['extendbox'])
        texture, elapsed_time_cls = self.classifier.inference(vol[np.newaxis], new_bbox, spacing)
        print(f'Nodule Texture Classification - Done with elapsed time = {elapsed_time_cls}')

        predictions = np.concatenate([scores, bbox, texture], axis=1)
        result = {
            "spacing": spacing.tolist(),
            "extendbox": data['extendbox'].tolist(),
            "top_five": predictions.tolist(),
            "message": "Successfully predicted.",
            "version": version_code,
            "valid": True,
            "elapsed_time_cls": elapsed_time_cls,
            "elapsed_time_det": elapsed_time_det,
            "elapsed_time_seg": elapsed_time_seg,
        }

        return result