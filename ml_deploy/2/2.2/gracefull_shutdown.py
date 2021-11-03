import os
import signal
import sys
import time

import redis
import requests

PORT = os.environ.get('PORT', 5000)
TIMEOUT = 15
replica_name = os.environ['REPLICA_NAME']
service_name = "web_app"


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        REDIS_IP = "95.216.168.158"
        REDIS_PORT = "6379"
        REDIS_PASSRORD = "lolkek123"
        self.rd = redis.Redis(host=REDIS_IP, port=REDIS_PORT,
                              password=REDIS_PASSRORD, decode_responses=True)

    def beat_check(self) -> None:
        service_replicas = self.rd.lrange(service_name, 0, -1)
        do_kill = replica_name not in service_replicas
        if do_kill:
            self.exit_gracefully()

    def unregister_service(self) -> None:
        """
        SIGKILL
        SIGINT - с возможностью перехвата
        SIGTERM - с возможностью перехвата
        """
        print("Unregister server")
        self.rd.lrem(service_name, 0, replica_name)
        all_keys = list(self.rd.hgetall(replica_name).keys())
        if all_keys:
            self.rd.hdel(replica_name, *all_keys)

    def exit_gracefully(self, *args):
        self.unregister_service()
        time.sleep(TIMEOUT)
        requests.post(f"http://localhost:{PORT}/shutdown")
        self.kill_now = True


if __name__ == '__main__':
    # fpid = os.fork()
    # print("fpid =", fpid)
    # if fpid != 0:
    #     # Running as daemon now. PID is fpid
    #     sys.exit(0)
    killer = GracefulKiller()
    while not killer.kill_now:
        time.sleep(0.3)
        print("Server beat...")
        # killer.beat_check()
    print("End of the program. I was killed gracefully :)")
    sys.exit(0)
