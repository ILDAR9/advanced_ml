import os
import signal
from itertools import cycle
from typing import List

from . import client_load_balancing, db

cur_service_name = None
replica_name = os.environ['REPLICA_NAME']
cur_host = None
cur_port = None


def register_service(service_name: str, host: str, port: int, expire_seconds=13) -> None:
    global cur_service_name, replica_name, cur_host, cur_port
    if not replica_name:
        raise RuntimeError("No replica_name")
    db.redis_connection.lpush(service_name, replica_name, )
    db.redis_connection.hset(replica_name, "host", host)
    db.redis_connection.hset(replica_name, "port", str(port))
    # db.redis_connection.expire(replica_name, expire_seconds)
    cur_service_name, cur_host, cur_port = service_name, host, port


def discover(service_name: str):
    def wrapped_discover() -> None:
        # notify
        # db.redis_connection.hset(replica_name, "host", cur_host)
        # db.redis_connection.hset(replica_name, "port", str(cur_port))

        replicas_pool = []
        print(service_name)
        service_replicas: List[str] = db.redis_connection.lrange(service_name, 0, -1)
        for replica in service_replicas:
            host, port = db.redis_connection.hmget(replica, keys=['host', 'port'])
            if (host is None) and (port is None):
                all_keys = list(db.redis_connection.hgetall(replica).keys())
                if all_keys:
                    db.redis_connection.hdel(replica, *all_keys)
                db.redis_connection.lrem(service_name, 0, replica)
            elif (host is not None) and (port is not None):
                replicas_pool.append(tuple([replica, host, port]))
            else:
                raise RuntimeError
        client_load_balancing.replicas_pool = cycle(replicas_pool)
        print(replicas_pool)

    return wrapped_discover
