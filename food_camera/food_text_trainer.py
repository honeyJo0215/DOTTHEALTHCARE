# food_text_trainer.py
r"""
텍스트 기반 음식 매칭 학습 + Vision API 테스트 스크립트

모드 1) 학습 (Vision API 안 씀, 무료)
    python food_text_trainer.py --train --translate-ko

모드 2) 테스트 (Vision API 사용, test_images 안 사진으로 테스트)
    python food_text_trainer.py --image test_images\fried_chicken.jpeg --topk 5

사전 준비:
    - food_db.csv 이미 생성되어 있어야 함 (name, serving_size, unit, calories, protein, fat, carbs)
    - Food-101 데이터는 이전에 이미지 학습 코드에서 다운로드해둔 상태라고 가정
      (없으면 한 번은 download=True로 Food101을 호출해서 받아둬야 함)
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

import argparse
import os
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd

from torchvision.datasets import Food101

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from google.cloud import vision

try:
    from googletrans import Translator
except ImportError:
    Translator = None


# ---------------- 경로 설정 ----------------
FOOD_DB_PATH = "food_db.csv"

VECTORIZER_PATH = "food_text_vectorizer.pkl"
FOOD_VECS_PATH = "food_db_tfidf.npz"
FOOD_META_PATH = "food_db_tfidf_meta.csv"
FOOD101_LABELS_PATH = "food101_labels_ko.csv"

MAX_VISION_LABELS = 10

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

FOOD101_MODEL_PATH = "food_classifier.pth"
FOOD101_CLASS_NAMES_PATH = "class_names.txt"

_food101_model = None
_food101_class_names = None


def load_food101_model():
    global _food101_model, _food101_class_names
    if _food101_model is not None:
        return _food101_model, _food101_class_names

    if not os.path.exists(FOOD101_MODEL_PATH):
        raise FileNotFoundError(f"{FOOD101_MODEL_PATH} 가 없습니다. Food-101 학습 모델이 필요합니다.")

    if not os.path.exists(FOOD101_CLASS_NAMES_PATH):
        raise FileNotFoundError(f"{FOOD101_CLASS_NAMES_PATH} 가 없습니다. class_names.txt를 확인하세요.")

    # class_names 로드
    with open(FOOD101_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    num_classes = len(class_names)

    # EfficientNet-B0 구조 재구성
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(FOOD101_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    _food101_model = model
    _food101_class_names = class_names
    return _food101_model, _food101_class_names


def predict_food101_labels(image_path: str, topk: int = 3):
    """
    로컬 Food-101 모델로 상위 topk 클래스 이름과 확률을 반환.
    return: (labels, probs)
        labels: ["fried chicken", "chicken wings", ...]
        probs:  [0.85, 0.07, ...]
    """
    model, class_names = load_food101_model()

    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    x = IMAGE_TRANSFORM(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        topk_probs, topk_idx = torch.topk(probs, k=min(topk, logits.shape[1]), dim=1)

    labels = []
    probs_list = topk_probs[0].tolist()
    idx_list = topk_idx[0].tolist()

    print("\n[Food-101 예측 라벨]")
    for rank, (idx, p) in enumerate(zip(idx_list, probs_list), start=1):
        raw_name = class_names[idx]  # 예: "fried_chicken"
        label_str = raw_name.replace("_", " ")
        labels.append(label_str)
        print(f"  {rank}. {label_str} (p={p:.3f})")

    return labels, probs_list

# ---------------- 공통 유틸 ----------------
def normalize_text(s: str) -> str:
    """검색/벡터라이저용 텍스트 정규화"""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("_", " ")
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

# 자주 나오는 쓸모없는 단어들(필터용)
CUSTOM_STOPWORDS = {
    "food", "foods", "recipe", "mix", "service", "company", "brand",
    "browning", "product", "products", "style",
    "vegetable", "vegetables", "fruit", "fruits"
}


def extract_keywords_from_labels(labels, max_keywords: int = 5) -> list[str]:
    """
    Vision 라벨 목록에서 의미 있는 키워드만 뽑아낸다.
    - 짧은 단어 제거 (len < 3)
    - stopword 제거 (food, mix, service, vegetable 등)
    - 중복 제거
    """
    # confidence 높은 순으로 정렬
    labels_sorted = sorted(labels, key=lambda x: x["score"], reverse=True)
    text = " ".join(l["description"] for l in labels_sorted)
    tokens = normalize_text(text).split()

    keywords: list[str] = []
    for t in tokens:
        if len(t) < 3:
            continue
        if t in CUSTOM_STOPWORDS:
            continue
        if t not in keywords:
            keywords.append(t)

    return keywords[:max_keywords]

def load_food_db(path: str = FOOD_DB_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. food_db를 먼저 생성해야 합니다.")

    df = pd.read_csv(path)

    need_cols = ["name", "serving_size", "unit", "calories", "protein", "fat", "carbs"]
    for col in need_cols:
        if col not in df.columns:
            if col in ["name", "unit"]:
                df[col] = ""
            else:
                df[col] = 0.0

    df = df[need_cols]
    return df


# ---------------- Vision API ----------------
def get_vision_labels(image_path: str, max_labels: int = MAX_VISION_LABELS) -> List[Dict]:
    """이미지에서 Google Vision API로 라벨 목록 추출"""
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image, max_results=max_labels)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    labels = []
    for label in response.label_annotations:
        labels.append({
            "description": label.description,
            "score": label.score,
        })

    print("\n[Vision Labels]")
    for l in labels:
        print(f"- {l['description']} ({l['score']:.3f})")

    return labels


# ---------------- 학습: TF-IDF + Food101 + (옵션) 한국어 번역 ----------------
def train_text_matcher(translate_ko: bool = False):
    """
    - food_db.csv의 name
    - Food-101 클래스 이름 (영어 + (옵션) 한국어 번역)
    를 이용해서 TF-IDF 벡터라이저를 학습하고,
    food_db 이름들을 벡터로 변환하여 저장.
    """
    print("📂 food_db 로딩 중...")
    df_food = load_food_db(FOOD_DB_PATH)

    # 1) Food-101 클래스 이름 읽기
    print("📂 Food-101 클래스 이름 로딩 중...")
    try:
        food101 = Food101(root="./data", split="train", download=False)
    except RuntimeError:
        # 데이터셋이 없으면 한 번은 직접 다운로드 필요
        print("⚠ Food-101 데이터셋이 없어서 download=False로는 로드 실패.")
        print("   -> 별도 스크립트에서 Food101(root='./data', split='train', download=True) 한 번 돌려주세요.")
        raise

    labels = food101.classes  # 예: ["apple_pie", "bibimbap", ...]
    label_records = []

    translator = None
    if translate_ko:
        if Translator is None:
            print("⚠ googletrans가 설치되어 있지 않아 영어만 사용합니다.")
        else:
            translator = Translator()
            print("🌐 googletrans 사용: Food-101 클래스명을 한국어로 번역 시도")

    corpus: List[str] = []

    # food_db의 이름도 코퍼스에 포함
    df_food["name_norm"] = df_food["name"].astype(str).apply(normalize_text)
    corpus.extend(df_food["name_norm"].tolist())

    # Food-101 클래스 이름도 코퍼스에 추가 (영어 + 한국어)
    for label in labels:
        display_en = label.replace("_", " ")
        display_en_norm = normalize_text(display_en)

        display_ko = ""
        if translator is not None:
            try:
                display_ko = translator.translate(display_en, src="en", dest="ko").text
            except Exception as e:
                print(f"  번역 실패: {display_en} -> {e}")
                display_ko = ""

        # 코퍼스에 추가
        corpus.append(display_en_norm)
        if display_ko:
            corpus.append(normalize_text(display_ko))

        label_records.append({
            "label_id": label,
            "display_en": display_en,
            "display_ko": display_ko,
        })

    # Food-101 라벨 ↔ 한글 매핑 파일 저장 (UI나 디버깅에 유용)
    pd.DataFrame(label_records).to_csv(FOOD101_LABELS_PATH, index=False, encoding="utf-8-sig")
    print(f"💾 Food-101 라벨 매핑 저장: {FOOD101_LABELS_PATH}")

    # 2) TF-IDF 벡터라이저 학습
    print("🧠 TF-IDF 벡터라이저 학습 중...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    vectorizer.fit(corpus)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"💾 벡터라이저 저장: {VECTORIZER_PATH}")

    # 3) food_db 이름들을 벡터로 변환해서 저장 (희소행렬 .npz)
    print("🔢 food_db 이름 TF-IDF 벡터화 중...")
    food_name_vecs = vectorizer.transform(df_food["name_norm"].tolist())
    sparse.save_npz(FOOD_VECS_PATH, food_name_vecs)
    print(f"💾 food_db TF-IDF 벡터 저장: {FOOD_VECS_PATH}")

    # 4) food_db 메타(영양정보)를 별도 저장 (인덱스 맞춰 사용)
    df_food[["name", "name_norm", "serving_size", "unit", "calories", "protein", "fat", "carbs"]].to_csv(
        FOOD_META_PATH, index=False, encoding="utf-8-sig"
    )

    print(f"💾 food_db 메타 저장: {FOOD_META_PATH}")

    print("✅ 텍스트 매칭 학습 완료!")


# ---------------- 테스트: Vision + TF-IDF 매칭 ----------------
def load_text_matcher():
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(FOOD_VECS_PATH) and os.path.exists(FOOD_META_PATH)):
        raise FileNotFoundError(
            "텍스트 매칭 모델 파일이 없습니다. 먼저 다음을 실행하세요:\n"
            "    python food_text_trainer.py --train"
        )

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    food_vecs = sparse.load_npz(FOOD_VECS_PATH)
    df_meta = pd.read_csv(FOOD_META_PATH)

    return df_meta, food_vecs, vectorizer


def predict_from_image(image_path: str, topk: int = 5):
    print(f"📷 테스트 이미지: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    # 0) (선택) Vision 라벨은 항상 받되, 나중에 fallback 용으로도 사용
    try:
        vision_labels = get_vision_labels(image_path)
    except Exception as e:
        print(f"⚠ Vision API 호출 실패 (무시하고 진행): {e}")
        vision_labels = []

    # 1) Food-101 분류기로 dish 이름 예측
    food101_labels, food101_probs = predict_food101_labels(image_path, topk=3)
    max_prob = food101_probs[0] if food101_probs else 0.0

    # 2) 텍스트 매칭 모델 로드
    print("📂 텍스트 매칭 모델 로딩 중...")
    df_meta, food_vecs, vectorizer = load_text_matcher()
    if "name_norm" not in df_meta.columns:
        df_meta["name_norm"] = df_meta["name"].astype(str).apply(normalize_text)

    # 3) 쿼리 텍스트와 키워드 결정 로직
    USE_FOOD101_THRESHOLD = 0.5  # 신뢰도 기준

    if max_prob >= USE_FOOD101_THRESHOLD:
        # Food-101 예측이 그나마 믿을 만하면 이걸 메인으로 사용
        print(f"\n[INFO] Food-101 예측 신뢰도 {max_prob:.3f} >= {USE_FOOD101_THRESHOLD}, Food-101 기반 검색 사용")
        query_text = " ".join(food101_labels)
        fake_label_objs = [{"description": t, "score": 1.0} for t in food101_labels]
        keywords = extract_keywords_from_labels(fake_label_objs, max_keywords=5)
    else:
        # Food-101가 자신 없으면 Vision 라벨 기반으로 fallback
        print(f"\n[INFO] Food-101 예측 신뢰도 {max_prob:.3f} < {USE_FOOD101_THRESHOLD}, Vision 기반 검색으로 fallback")
        # Vision 라벨 텍스트 합치기
        labels_sorted = sorted(vision_labels, key=lambda x: x["score"], reverse=True)
        query_text = " ".join(l["description"] for l in labels_sorted)
        keywords = extract_keywords_from_labels(vision_labels, max_keywords=5)

    query_norm = normalize_text(query_text)
    print(f"\n[검색 쿼리 텍스트]\n  {query_text}")
    print(f"[정규화 텍스트]\n  {query_norm}")
    print(f"[추출된 키워드] {keywords}")

    # 4) 키워드가 들어있는 food_db 후보만 우선 검색
    if keywords:
        mask = df_meta["name_norm"].fillna("").apply(
            lambda s: any(k in s for k in keywords)
        )
        df_cand = df_meta[mask].copy()
        food_vecs_cand = food_vecs[mask.values]
        if len(df_cand) == 0:
            print("⚠ 키워드로 매칭되는 음식이 없어 전체 food_db에서 검색합니다.")
            df_cand = df_meta.copy()
            food_vecs_cand = food_vecs
    else:
        df_cand = df_meta.copy()
        food_vecs_cand = food_vecs

    # 5) 코사인 유사도
    q_vec = vectorizer.transform([query_norm])
    sims = cosine_similarity(q_vec, food_vecs_cand).flatten()

    if topk > len(sims):
        topk = len(sims)
    top_idx = np.argpartition(-sims, range(topk))[:topk]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    df_top = df_cand.iloc[top_idx].copy()
    df_top["match_score"] = sims[top_idx]

    print("\n🍽 추천 음식 후보 (상위 {}개):".format(topk))
    for i, row in enumerate(df_top.itertuples(), start=1):
        print(f"\n[{i}] {row.name}  (유사도: {row.match_score:.3f})")
        print(f"    - 기준량: {row.serving_size} {row.unit}")
        print(f"    - 칼로리: {row.calories} kcal")
        print(f"    - 탄수화물: {row.carbs} g")
        print(f"    - 단백질: {row.protein} g")
        print(f"    - 지방: {row.fat} g")


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="텍스트 매칭 학습 모드 (Vision API 사용 안 함)")
    parser.add_argument("--translate-ko", action="store_true", help="학습 시 Food-101 이름을 한국어로 번역해서 코퍼스에 포함")
    parser.add_argument("--image", type=str, help="Vision API + 텍스트 매칭으로 테스트할 이미지 경로")
    parser.add_argument("--topk", type=int, default=5, help="상위 몇 개 음식 후보를 보여줄지")

    args = parser.parse_args()

    if args.train:
        train_text_matcher(translate_ko=args.translate_ko)
    elif args.image:
        predict_from_image(args.image, topk=args.topk)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
