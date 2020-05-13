# project/server/__init__.py


import os

from flask import Flask
from flask_bootstrap import Bootstrap

# instantiate the extensions
bootstrap = Bootstrap()


def create_app(script_info=None):

    # instantiate the app
    app = Flask(
        __name__,
        template_folder="../client/templates",
        static_folder="../client/static",
    )

    # set config
    app_settings = os.getenv("APP_SETTINGS")
    app.config.from_object(app_settings)

    # set up extensions
    bootstrap.init_app(app)

    # register blueprints
    from project.server.main.views import main_blueprint

    app.register_blueprint(main_blueprint)

    # shell context for flask cli
    app.shell_context_processor({"app": app})

    return app

def create_worker_app():

    # instantiate the app
    app = Flask(__name__)

    # set config
    app_settings = os.getenv("APP_SETTINGS")
    app.config.from_object(app_settings)

    from project.engine import NoduleDetectionInference

    detector = NoduleDetectionInference(
        config_path='models/nodule_detection_yi_s3_b210/model.yaml',
        pretrain_weights=['models/nodule_detection_yi_s3_b210/weight.ckpt'],
        gpu_device=0
    )

    app.config['detector'] = detector

    return app