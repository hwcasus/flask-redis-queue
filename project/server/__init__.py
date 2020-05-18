import os
import torch
from flask import Flask

def create_app(script_info=None):

    # instantiate the app
    app = Flask(__name__,)

    # set config
    app_settings = os.getenv("APP_SETTINGS")
    app.config.from_object(app_settings)

    # register blueprints
    from project.server.main.views import main_blueprint
    app.register_blueprint(main_blueprint)

    return app

def create_worker_app():

    # instantiate the app
    app = Flask(__name__)

    # set config
    app_settings = os.getenv("APP_SETTINGS")
    app.config.from_object(app_settings)

    from project.engine import NoduleDetectionInference, LungSegmentationInference, NoduleClassifyInference

    app.config['detector'] = NoduleDetectionInference(
        config_path='models/nodule_detection_yi_s3_b210/model.yaml',
        pretrain_weights=['models/nodule_detection_yi_s3_b210/weight.ckpt'],
        gpu_device=0
    )
    app.config['segmentor'] = LungSegmentationInference(
        config_path='models/lung_segmentation/model_revised.yaml',
        pretrain_weights=['models/lung_segmentation/weight.ckpt'],
        gpu_device=1
    )
    app.config['classifier'] = NoduleClassifyInference(
        config_path='models/nodule_classification/model_concat.yaml',
        pretrain_weights=['models/nodule_classification/weight.ckpt'],
        gpu_device=2
    )

    return app
