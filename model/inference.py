import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# 저장된 Fine-tuned 모델과 토크나이저 불러오기
#model_dir = 'hyun5oo/hansoldeco'
model_dir = "llama-2-7b-hansol-deco-eos"
model = AutoModelForCausalLM.from_pretrained(model_dir, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Inference를 위한 test.csv 파일 로드
test = pd.read_csv('./test.csv')

# test.csv의 '질문'에 대한 '답변'을 저장할 리스트
preds = []

preds = []
# '질문' 컬럼의 각 질문에 대해 답변 생성
for test_question in tqdm(test['질문']):
    # 입력 텍스트를 토큰화하고 모델 입력 형태로 변환
    input_ids = tokenizer.encode(test_question + tokenizer.eos_token, return_tensors='pt')

    # 답변 생성
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=600,
        temperature=0.9,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )

    # 생성된 텍스트(답변) 저장
    for generated_sequence in output_sequences:
        full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
        # 질문과 답변의 사이를 나타내는 eos_token (</s>)를 찾아, 이후부터 출력
        answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
        answer_only = full_text[answer_start:].strip()
        answer_only = answer_only.replace('\n', ' ')
        preds.append(answer_only)

    # Test 데이터셋의 모든 질의에 대한 답변으로부터 512 차원의 Embedding Vector 추출
    # 평가를 위한 Embedding Vector 추출에 활용하는 모델은 'distiluse-base-multilingual-cased-v1' 이므로 반드시 확인해주세요.
    from sentence_transformers import SentenceTransformer  # SentenceTransformer Version 2.2.2

    # Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # 생성한 모든 응답(답변)으로부터 Embedding Vector 추출
    pred_embeddings = model.encode(preds)
    pred_embeddings.shape

    submit = pd.read_csv('./sample_submission.csv')
    # 제출 양식 파일(sample_submission.csv)을 활용하여 Embedding Vector로 변환한 결과를 삽입
    submit.iloc[:, 1:] = pred_embeddings
    submit.head()

    # 리더보드 제출을 위한 csv파일 생성
    submit.to_csv('./baseline_submit.csv', index=False)