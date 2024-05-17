# import re
# from transformers import BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import train_test_split
# import torch
# import joblib
# import os

# # 텍스트 전처리 함수
# def preprocess_text(text):
#     text = re.sub(r'\d+', '', text)  # 숫자 제거
#     text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
#     text = text.lower()  # 소문자 변환
#     return text

# # BERT 모델과 토크나이저 로드
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertModel.from_pretrained('bert-base-multilingual-cased')

# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# def calculate_similarity(text1, text2):
#     embedding1 = get_embedding(text1).reshape(1, -1)
#     embedding2 = get_embedding(text2).reshape(1, -1)
#     similarity = cosine_similarity(embedding1, embedding2)
#     return similarity[0][0]

# def recommend_career(student_text, job_descriptions):
#     recommendations = []
#     for job, description in job_descriptions.items():
#         similarity = calculate_similarity(student_text, description)
#         recommendations.append((job, similarity))
#     recommendations.sort(key=lambda x: x[1], reverse=True)
#     return recommendations

# # 직업 설명 데이터
# job_descriptions = {
#     "국회의원": "국회의원은 선거를 통해 선출된 국민의 대표로서 국회에서 헌법과 법률의 개정 및 의결과 관련된 일을 하고, 정부 예산안을 심의 및 확정하는 등의 업무를 담당합니다.",
#     "행정부고위공무원": "행정부고위공무원은 행정기관 국장급(3급) 이상 공무원으로, 정부의 정책을 결정하고 예산과 법령안을 작성·수정하며 정부 부처의 법령을 해석·적용합니다.",
#     "기업고위임원": "기업고위임원은 경영학적 지식을 바탕으로 기업의 기본 경영방침과 목표를 계획하고, 목표를 달성하기 위한 전략과 정책을 수립합니다.",
#     "외교관": "외교관은 본국을 대표하여 외국에 파견되어 외국과의 교섭을 통해 정치, 경제, 상업적 이익의 보호와 증진을 추구하며, 해외동포와 해외여행을 하는 자국민을 보호합니다.",
#     "교장": "교장은 초등학교나 중학교, 고등학교, 특수학교를 대표하는 책임자로서 학교의 교육, 행정 및 기타 운영 활동에 관련된 모든 제반 사항을 기획하고 조정하는 일을 합니다.",
#     "호텔지배인": "호텔지배인은 객실 예약과 판매, 고객 안내, 식당 운영, 호텔 홍보 등 호텔에서 이루어지는 다양한 업무들이 잘 운영될 수 있도록 각종 활동을 계획하고, 호텔 종사원의 업무를 종합적으로 관리 및 감독하는 일을 합니다.",
#     "영업원": "영업원은 아직 제품이나 서비스에 대해 구매 의사가 없는 사람이 구매를 할 수 있도록 권하거나 판매합니다.",
#     # 기타 직업 설명 추가
# }

# # 예시 데이터
# student_records = [
#     {
#         "name": "김철수",
#         "text": "김철수 학생은 학습에 대한 열의가 높고, 특히 과학 과목에서 우수한 성과를 보였습니다. 실험 과정에서 창의적인 아이디어를 제시하고, 문제 해결 능력이 뛰어납니다. 또한, 학교 축제에서 주도적인 역할을 하며, 친구들과의 협력도 잘 이루어졌습니다.",
#         "career_label": "과학교사"
#     },
#     {
#         "name": "박영희",
#         "text": "박영희 학생은 문학과 예술에 큰 관심을 가지고 있으며, 국어 과목에서 뛰어난 독해력과 비판적 사고 능력을 보여주었습니다. 다양한 문학 작품을 깊이 있게 분석하고, 창의적인 글쓰기에 능합니다. 또한, 학교 신문 동아리에서 활동하며, 뛰어난 글쓰기 실력을 발휘하고 있습니다.",
#         "career_label": "국어교사"
#     }
#     # 추가 학생 기록...
# ]

# # 직업 라벨과 텍스트를 추출합니다.
# texts = [preprocess_text(record['text']) for record in student_records]
# labels = [record['career_label'] for record in student_records]

# # 학습 데이터와 테스트 데이터로 분리합니다.
# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# def train_and_evaluate_model(X_train, y_train, X_test, y_test, job_descriptions):
#     predictions = []
#     for text in X_test:
#         recommended_careers = recommend_career(text, job_descriptions)
#         predictions.append(recommended_careers[0][0])  # 유사도가 가장 높은 직업 선택

#     accuracy = accuracy_score(y_test, predictions)
#     precision = precision_score(y_test, predictions, average='macro', zero_division=0)
#     recall = recall_score(y_test, predictions, average='macro', zero_division=0)
#     f1 = f1_score(y_test, predictions, average='macro', zero_division=0)

#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"F1 Score: {f1}")

# # 모델 학습 및 평가
# train_and_evaluate_model(X_train, y_train, X_test, y_test, job_descriptions)

# # 모델과 토크나이저 저장
# os.makedirs('hackathon_fastpai/models', exist_ok=True)
# joblib.dump(tokenizer, os.path.join('hackathon_fastpai/models', 'tokenizer.joblib'))
# torch.save(model.state_dict(), os.path.join('hackathon_fastpai/models', 'bert_model.pth'))
# torch.save(model, os.path.join('hackathon_fastpai/models', 'bert_model_full.pth'))  # 전체 모델 저장 (옵션)
# 2. Flask 서버 코드 (app.py)
# Flask 서버에서도 동일한 경로를 사용하여 모델과 토크나이저를 로드합니다.

# python
# 코드 복사
# from flask import Flask, request, jsonify
# import re
# import torch
# import joblib
# from transformers import BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity
# import fitz  # PyMuPDF
# from konlpy.tag import Okt
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = './uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # 텍스트 전처리 함수
# def preprocess_text(text):
#     text = re.sub(r'\d+', '', text)  # 숫자 제거
#     text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
#     text = text.lower()  # 소문자 변환
#     return text

# # 모델과 토크나이저 로드
# tokenizer = joblib.load(os.path.join('hackathon_fastpai/models', 'tokenizer.joblib'))
# model = BertModel.from_pretrained('bert-base-multilingual-cased')
# model.load_state_dict(torch.load(os.path.join('hackathon_fastpai/models', 'bert_model.pth')))
# model.eval()

# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# def calculate_similarity(text1, text2):
#     embedding1 = get_embedding(text1).reshape(1, -1)
#     embedding2 = get_embedding(text2).reshape(1, -1)
#     similarity = cosine_similarity(embedding1, embedding2)
#     return similarity[0][0]

# def recommend_career(student_text, job_descriptions):
#     recommendations = []
#     for job, description in job_descriptions.items():
#         similarity = calculate_similarity(student_text, description)
#         recommendations.append((job, similarity))
#     recommendations.sort(key=lambda x: x[1], reverse=True)
#     return recommendations

# # 예시 직업 설명 데이터
# job_descriptions = {
#     "국회의원": "국회의원은 선거를 통해 선출된 국민의 대표로서 국회에서 헌법과 법률의 개정 및 의결과 관련된 일을 하고, 정부 예산안을 심의 및 확정하는 등의 업무를 담당합니다.",
#     "행정부고위공무원": "행정부고위공무원은 행정기관 국장급(3급) 이상 공무원으로, 정부의 정책을 결정하고 예산과 법령안을 작성·수정하며 정부 부처의 법령을 해석·적용합니다.",
#     "기업고위임원": "기업고위임원은 경영학적 지식을 바탕으로 기업의 기본 경영방침과 목표를 계획하고, 목표를 달성하기 위한 전략과 정책을 수립합니다.",
#     "외교관": "외교관은 본국을 대표하여 외국에 파견되어 외국과의 교섭을 통해 정치, 경제, 상업적 이익의 보호와 증진을 추구하며, 해외동포와 해외여행을 하는 자국민을 보호합니다.",
#     "교장": "교장은 초등학교나 중학교, 고등학교, 특수학교를 대표하는 책임자로서 학교의 교육, 행정 및 기타 운영 활동에 관련된 모든 제반 사항을 기획하고 조정하는 일을 합니다.",
#     "호텔지배인": "호텔지배인은 객실 예약과 판매, 고객 안내, 식당 운영, 호텔 홍보 등 호텔에서 이루어지는 다양한 업무들이 잘 운영될 수 있도록 각종 활동을 계획하고, 호텔 종사원의 업무를 종합적으로 관리 및 감독하는 일을 합니다.",
#     "영업원": "영업원은 아직 제품이나 서비스에 대해 구매 의사가 없는 사람이 구매를 할 수 있도록 권하거나 판매합니다.",
#     # 기타 직업 설명 추가
# }

# def extract_text_from_pdf(pdf_path):
#     # PDF 문서 열기
#     document = fitz.open(pdf_path)
#     text = ""

#     # 각 페이지의 텍스트 추출
#     for page_num in range(document.page_count):
#         page = document[page_num]
#         text += page.get_text()

#     return text

# def extract_nouns(text):
#     # Remove line breaks
#     text = text.replace('\n', ' ')
    
#     # Initialize Okt tokenizer
#     okt = Okt()
    
#     # Extract nouns
#     nouns = okt.nouns(text)
    
#     return ' '.join(nouns)

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.json
#     student_text = preprocess_text(data['text'])
#     recommended_careers = recommend_career(student_text, job_descriptions)
#     return jsonify({"name": data['name'], "recommendations": [job for job, _ in recommended_careers[:3]]})

# @app.route('/upload_pdf', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return "No file part", 400
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file", 400
    
#     # PDF 파일 저장
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(file_path)

#     # PDF 텍스트 추출
#     extracted_text = extract_text_from_pdf(file_path)
#     extracted_nouns = extract_nouns(extracted_text)
#     student_text = preprocess_text(extracted_nouns)

#     recommended_careers = recommend_career(student_text, job_descriptions)
#     recommendations_text = ', '.join([job for job, _ in recommended_careers[:3]])

#     # 결과를 Spring 서버로 전송 (예시)
#     # requests.post('http://your-spring-server-url/recommendations', json={"name": file.filename.split('.')[0], "recommendations": recommendations_text})

#     return jsonify({"name": file.filename.split('.')[0], "recommendations": recommendations_text})

# if __name__ == '__main__':
#     app.run(debug=True)
