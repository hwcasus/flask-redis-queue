from .lung_segment import LungSegmentationInference
from .nodule_classify import NoduleClassifyInference
from .nodule_detect import NoduleDetectionInference
from .nodule_detect_body import NoduleDetectionBodyInference
from .nodule_segment_classify import NoduleSegClassifyInference


TOP_K = 5
VERSION_CODE = "0417"

# detector = NoduleDetectionBodyInference(
#     config_path='models/nodule_detection_fbody_closing/model.yaml',
#     pretrain_weights=['models/nodule_detection_fbody_closing/weight.ckpt'],
#     gpu_device=0
# )
# detector = NoduleDetectionInference(
#     config_path='models/nodule_detection_yi_s3_b210/model.yaml',
#     pretrain_weights=['models/nodule_detection_yi_s3_b210/weight.ckpt'],
#     gpu_device=0
# )
# segmentor = LungSegmentationInference(
#     config_path='models/lung_segmentation/model_revised.yaml',
#     pretrain_weights=['models/lung_segmentation/weight.ckpt'],
#     gpu_device=1
# )
# classifier = NoduleClassifyInference(
#     config_path='models/nodule_classification/model_concat.yaml',
#     pretrain_weights=['models/nodule_classification/weight.ckpt'],
#     gpu_device=2
# )
# classifier = NoduleSegClassifyInference(
#     config_path='models/nodule_segment_classification/model.yaml',
#     pretrain_weights=['models/nodule_segment_classification/weight.ckpt'],
#     gpu_device=2
# )


# def inference_task(input_data):
#     series_list = input_data['series']
#     detail = [
#         (series['series_uid'], series['image_url'])
#         for series in series_list
#     ]
#     print(detail)
#     return detail
