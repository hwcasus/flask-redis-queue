import os
import time
import json

import numpy as np
import requests
from flask import current_app

from project.utils import load_dcm

TOP_K = 5
VERSION_CODE = "0417"


def inference_task(input_dict):
    response = {}
    post_api = input_dict['post_api']
    for series in input_dict['series']:
        series_id = series['series_uid']
        image_url_list = series['image_url']
        sub_dir = f'tmp/{series_id}'
        save_dir = os.path.join(current_app.instance_path, sub_dir)
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

    response_pickled = json.dumps(response)
    headers = {'content-type': 'application/json'}
    r = requests.post(post_api, headers=headers, data=response_pickled)
