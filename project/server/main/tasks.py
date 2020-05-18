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

        result = current_app.config['InferencePipeline'].inference(vol, spacing)
        response[series_id].update(result)

    response_pickled = json.dumps(response)
    headers = {'content-type': 'application/json'}
    r = requests.post(post_api, headers=headers, data=response_pickled)
