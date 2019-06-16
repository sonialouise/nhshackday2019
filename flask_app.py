import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import ImageClassifier

app = Flask(__name__)
CORS(app)


@app.route("/infected/", methods=['GET'])
def return_prediction():
  image = request.args.get('image')
  infected = ImageClassifier.predict(input_image=image)
  infected_dict = {
                'model':'mlp',
                'infected': infected,
                }
  return jsonify(infected_dict)


@app.route("/",methods=['GET'])
def default():
  return "<h1> Welcome to bitcoin price predictor <h1>"


if __name__ == "__main__":
    app.run()
