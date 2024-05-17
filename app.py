from flask import Flask, request, jsonify
import re
import torch
import joblib
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from konlpy.tag import Okt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = text.lower()  # 소문자 변환
    return text

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1).reshape(1, -1)
    embedding2 = get_embedding(text2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

def recommend_career(student_text, job_descriptions):
    recommendations = []
    for job, description in job_descriptions.items():
        similarity = calculate_similarity(student_text, description)
        recommendations.append((job, similarity))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# 예시 직업 설명 데이터
job_descriptions = {
    "국회의원": "국회의원은 선거를 통해 선출된 국민의 대표로서 국회에서 헌법과 법률의 개정 및 의결과 관련된 일을 하고, 정부 예산안을 심의 및 확정하는 등의 업무를 담당합니다.",
    "행정부고위공무원": "행정부고위공무원은 행정기관 국장급(3급) 이상 공무원으로, 정부의 정책을 결정하고 예산과 법령안을 작성·수정하며 정부 부처의 법령을 해석·적용합니다.",
    "기업고위임원": "기업고위임원은 경영학적 지식을 바탕으로 기업의 기본 경영방침과 목표를 계획하고, 목표를 달성하기 위한 전략과 정책을 수립합니다.",
    "외교관": "외교관은 본국을 대표하여 외국에 파견되어 외국과의 교섭을 통해 정치, 경제, 상업적 이익의 보호와 증진을 추구하며, 해외동포와 해외여행을 하는 자국민을 보호합니다.",
    "교장": "교장은 초등학교나 중학교, 고등학교, 특수학교를 대표하는 책임자로서 학교의 교육, 행정 및 기타 운영 활동에 관련된 모든 제반 사항을 기획하고 조정하는 일을 합니다.",
    "호텔지배인": "호텔지배인은 객실 예약과 판매, 고객 안내, 식당 운영, 호텔 홍보 등 호텔에서 이루어지는 다양한 업무들이 잘 운영될 수 있도록 각종 활동을 계획하고, 호텔 종사원의 업무를 종합적으로 관리 및 감독하는 일을 합니다.",
    "영업원": "영업원은 아직 제품이나 서비스에 대해 구매 의사가 없는 사람이 구매를 할 수 있도록 권하거나 판매합니다.",
    # 기타 직업 설명 추가
}

def extract_text_from_pdf(pdf_path):
    # PDF 문서 열기
    document = fitz.open(pdf_path)
    text = ""

    # 각 페이지의 텍스트 추출
    for page_num in range(document.page_count):
        page = document[page_num]
        text += page.get_text()

    return text

def extract_nouns(text):
    # Remove line breaks
    text = text.replace('\n', ' ')
    
    # Initialize Okt tokenizer
    okt = Okt()
    
    # Extract nouns
    nouns = okt.nouns(text)
    
    return ' '.join(nouns)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    student_text = preprocess_text(data['text'])
    recommended_careers = recommend_career(student_text, job_descriptions)
    return jsonify({"name": data['name'], "recommendations": [job for job, _ in recommended_careers[:3]]})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # PDF 파일 저장
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # PDF 텍스트 추출
    extracted_text = extract_text_from_pdf(file_path)
    extracted_nouns = extract_nouns(extracted_text)  # 함수 호출 추가
