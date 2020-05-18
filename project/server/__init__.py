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

    from project.engine import InferencePipeline
    app.config['InferencePipeline'] = InferencePipeline(top_k=5, version_code='0417')

    return app
