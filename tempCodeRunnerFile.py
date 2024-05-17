import os

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/', memthods=['GET'])
def index():
    return jsonify({'message': 'Welcome to the Flask server!'})

@app.route('/receive_pdf', methods=['POST'])
def receive_pdf():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']

    # PDF 파일을 저장할 디렉터리 생성
    upload_dir = '/Users/yerong/study/hackathon-flask-pdf'

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # 파일을 저장
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    return jsonify({'message': 'File received successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
