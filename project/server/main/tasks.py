# project/server/main/tasks.py


from flask import current_app
import time


def create_task(task_type):
    time.sleep(int(task_type) * 10)
    return id(current_app.config['detector'])
