from flask import Blueprint, jsonify, request

from . import db

bp_sd = Blueprint(name="sd_blueprint",
                  import_name=__name__,
                  url_prefix="/"
                  )


@bp_sd.route("/return_secret_number", methods=["GET"])
def return_secret_number():
    return jsonify({"secret_number": db.secret_number})
