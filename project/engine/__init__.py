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

def inference_task(input_dict):
    for series in input_dict['series']:
        series_id = series['series_uid']
        image_url_list = series['image_url']
        sub_dir = f'tmp/{series_id}'
        save_dir = os.path.join(app.instance_path, sub_dir)
        os.makedirs(save_dir, exist_ok=True)

        print("Process start")

        # Download image
        for idx, image_url in enumerate(image_url_list):
            img = requests.get(image_url)
            file_name = image_url.rsplit('/')[-1]
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, 'wb') as f:
                f.write(img.content)
            print(f"Image {idx}/{len(image_url_list)} retrieved")

        # Load set of dicom file into numpy array
        files = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
        vol, spacing = load_dcm(files)
        response[series_id] = dict()
        print('Data Loading - Done')

        if vol is None or spacing is None:
            response[series_id].update({
                "message": 'Invalid series with z-spacing greater than 2.5mm.',
                "valid": False
            })
            continue

        # Notive that this is a workaround due to reversed image list from server
        vol = np.flip(vol, 0)

        # Segmentation
        output, elapsed_time_seg = current_app.config['segmentor'].inference(vol[np.newaxis])
        mask = output['mask'].squeeze()
        print(f'Lung Segmentation - Done with elapsed time = {elapsed_time_seg}')

        # Detection
        (ret, data), elapsed_time_det = current_app.config['detector'].inference(vol, mask, spacing)
        pred = np.array([bbox.cpu().numpy() for bbox in ret['bbox']]).squeeze()  # [50, 5], [score, z, y, x, d]
        print(f'Lung Nodule Detection - Done with elapsed time = {elapsed_time_det}')

        # Re-order prediction by z-axis coordinate
        pred = pred[np.argsort(pred[:, 1])]

        # Filter by threshold
        # threshold for yi_s3_b210
        # threshold = 0.926097542 #  0.980911999
        threshold = 0.980911999
        # threshold for fbody_closing
        # threshold = 0.972237150 #  0.993234332
        pred = pred[pred[:, 0] > threshold]

        # Select Top K (Default: 10)
        scores = pred[:TOP_K, :1]
        bbox = pred[:TOP_K, 1:]

        # Label resampling, back to original spacing
        new_bbox = current_app.config['detector'].label_resampling(bbox, spacing, extendbox=data['extendbox'])

        # Classification
        ret, elapsed_time_cls = current_app.config['classifier'].inference(vol[np.newaxis], new_bbox, spacing)
        texture_prob = ret.cpu().numpy()
        texture = texture_prob.argmax(1)[:, np.newaxis]
        predictions = np.concatenate([scores, bbox, texture], axis=1)
        print(f'Nodule Texture Classification - Done with elapsed time = {elapsed_time_cls}')

        response[series_id].update({
            "spacing": spacing.tolist(),
            "extendbox": data['extendbox'].tolist(),
            "top_five": predictions.tolist(),
            "message": "Successfully predicted.",
            "version": VERSION_CODE,
            "valid": True,
            "elapsed_time_cls": elapsed_time_cls,
            "elapsed_time_det": elapsed_time_det,
            "elapsed_time_seg": elapsed_time_seg,
        })

        if draw:
            # Drawing bbox on volume for debug, shouldn't exist when actually deployed
            result_dir = os.path.join(app.instance_path, 'results')
            os.makedirs(result_dir, exist_ok=True)

            # Draw output from detector on input (preprocessed) volume
            new_bboxes = np.round(bbox).astype(int)
            new_vol_resampled = np.repeat(data['vol_resampled'], 3, axis=0)
            draw_bbox(new_vol_resampled, new_bboxes[:5])

            image_list = [s for s in new_vol_resampled.transpose((1, 2, 3, 0))]
            imageio.mimsave(os.path.join(result_dir, 'det.gif'), image_list)

            # Label resampling, back to original spacing
            new_bboxes = current_app.config['detector'].label_resampling(bbox, spacing, extendbox=data['extendbox'])

            # Draw resampled output on original volume
            new_vol = np.repeat(vol[np.newaxis], 3, axis=0)
            new_vol = (new_vol - new_vol.min()) / (new_vol.max() - new_vol.min()) * 255
            draw_bbox(new_vol, new_bboxes[:5])

            image_list = [s for s in new_vol.transpose((1, 2, 3, 0))]
            imageio.mimsave(os.path.join(result_dir, 'det_origin.gif'), image_list)

            image_list = [s for s in mask[..., np.newaxis]]
            imageio.mimsave(os.path.join(result_dir, 'mask.gif'), image_list)

    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")