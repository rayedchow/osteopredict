from flask import Flask, request
from model import predictRisk
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=['POST'])
@cross_origin()
def risk():
    data = request.get_json()
    predictedRisk = predictRisk(data)
    return predictedRisk
