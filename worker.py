import redis
from rq import Connection, Worker

from project.server import create_app


app = create_app()

def run_worker():
    redis_url = app.config["REDIS_URL"]
    redis_connection = redis.from_url(redis_url)
    with Connection(redis_connection):
        worker = Worker(app.config["QUEUES"])
        worker.work()


if __name__ == "__main__":
    run_worker()
