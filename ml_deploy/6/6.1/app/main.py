import os
from flask import Flask, jsonify, request

import db


app = Flask(__name__)


def receive_secret_key():
    db.secret_number = os.environ['SECRET_NUMBER']

@app.route("/return_secret_number", methods=["GET"])
def return_secret_number():
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
