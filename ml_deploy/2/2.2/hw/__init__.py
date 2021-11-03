import time
from os import sched_get_priority_max

import redis
import requests
from flask_apscheduler import APScheduler

from . import db, service_discover
from .service_discover import register_service

SECRET_URL = "https://lab.karpov.courses/hardml-api/module-5/get_secret_number"

def receive_secret_key():
    while True:
        response = requests.get(SECRET_URL)
        if response.status_code == 200:
            db.secret_number = response.json()['secret_number']
            break
        else:
            time.sleep(0.3)
            

def init_redis(host: str, port: str, password: str):
    if db.redis_connection is None:
        db.redis_connection = redis.Redis(
            host, port=port, password=password, decode_responses=True)
    else:
        pass


def init_sheduler(service_name: str):
    service_discover.discover(service_name=service_name)
    scheduler = APScheduler()
    scheduler.add_job(id="kek",
                      func=service_discover.discover(
                          service_name=service_name),
                      trigger='interval', seconds=5
                      )
    return scheduler
