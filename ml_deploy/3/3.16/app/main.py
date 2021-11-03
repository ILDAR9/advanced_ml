import os
import time

import requests
from flask import Flask, jsonify, request

import db

TIMEOUT = 1

app = Flask(__name__)

SECRET_URL = "https://lab.karpov.courses/hardml-api/module-5/get_secret_number"

def receive_secret_key():
    while True:
        response = requests.get(SECRET_URL)
        if response.status_code == 200:
            db.secret_number = response.json()['secret_number']
            break
        else:
            time.sleep(0.3)

@app.route("/return_secret_number", methods=["GET"])
def return_secret_number():
    time.sleep(TIMEOUT)
    return jsonify({"secret_number": db.secret_number})


@app.route("/info", methods=["GET"])
def info():
    pid = os.getpid()
    return f"PID={pid} host={request.host}"

if db.secret_number is None:
    receive_secret_key()


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 80
    print("Flask Current port is", PORT)
    print("Flask PID", os.getpid())
    app.run(host=HOST, port=PORT)
