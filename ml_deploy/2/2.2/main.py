import os

from flask import Flask, request

from hw import init_redis, init_sheduler, receive_secret_key, register_service
from hw import view as sd_view

app = Flask(__name__)

def shutdown_server():
    print("Server shutting down...")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route("/hello")
def hello_world():
    pid = os.getpid()
    return f"Hello world! PID={pid} host={request.host}"


if __name__ == "__main__":
    # configs
    IP1 = "95.216.168.89"
    IP2 = "95.216.168.158"
    REDIS_PASSRORD = "lolkek123"
    REDIS_PORT = "6379"
    REDIS_IP = IP2
    SERVICE_NAME = "web_app"

    HOST = "0.0.0.0"

    PORT = os.environ.get('PORT', 5000)
    print("Current port is", PORT)
    print("PID", os.getpid())

    receive_secret_key()
    init_redis(host=REDIS_IP, port=REDIS_PORT, password=REDIS_PASSRORD)
    register_service(service_name=SERVICE_NAME, host=IP1, port=PORT)
    scheduler = init_sheduler(service_name=SERVICE_NAME)
    scheduler.init_app(app)
    scheduler.start()

    app.register_blueprint(sd_view.bp_sd)
    app.run(host=HOST, port=PORT)
