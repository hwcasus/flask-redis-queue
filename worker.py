import redis
import torch
from rq import Connection, SimpleWorker

from project.server import create_worker_app


app = create_worker_app()


def run_worker():
    redis_url = app.config["REDIS_URL"]
    redis_connection = redis.from_url(redis_url)
    with Connection(redis_connection):
        worker = SimpleWorker(app.config["QUEUES"])
        worker.work()


if __name__ == "__main__":

    with app.app_context():
        run_worker()
