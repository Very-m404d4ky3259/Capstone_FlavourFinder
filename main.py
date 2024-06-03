from flask import Flask, request, jsonify, make_response
from helper import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_get():
    response = make_response(jsonify({
        'status': 'Success!',
        'message': 'FlavourFinder Model Ready!'
    }))
    response.status_code = 200
    response.headers['Content-Type'] = 'application/json'

    return response.json

@app.route('/classify', methods=['POST'])
def classify():
    try: 
        json_response = request.get_json()
        classification_result, percentage = classification(json_response["payload"])
       
        response_data = {
            'status': 'Success!',
            'message': classification_result,
            'percentage': str("{}%".format(round(percentage)))
        }
        response = make_response(jsonify(response_data))
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        
        return response
    except Exception as e:
        response = make_response(jsonify({
            'status': 'Error',
            'message': str(e)
        }))
        response.status_code = 500
        response.headers['Content-Type'] = 'application/json'

        return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)


# Notes before deployment
# 1. Enable The https://console.cloud.google.com/apis/library/iam.googleapis.com
# 5. Make Sure You Have The Necessary Permission To Perform The Deployment
# 2. Make Sure The RAM Is Set to 2.0 GB
# 3. Make Sure You Do Not Specify the Exact Version Of The Tensorflow in your requirements.txt
# 4. Set the region to Jakarta and not any other region (optional)
# 5. Git Naming Conventions That You Can Try (https://dev.to/ishanmakadia/git-commit-message-convention-that-you-can-follow-1709)