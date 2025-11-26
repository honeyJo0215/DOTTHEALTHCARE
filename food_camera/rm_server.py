# server.py
import os
import io
import json
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import Food101
from PIL import Image
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle

# 번역 / Vision 은 옵션
try:
    from googletrans import Translator
except ImportError:
    Translator = None

try:
    from google.cloud import vision
except ImportError:
    vision = None

# ---------------- 기본 설정 ----------------

MODEL_PATH = "food_classifier.pth"
CLASS_NAMES_PATH = "class_names.txt"
CUSTOM_DATA_DIR = "./custom_data"   # 사용자가 피드백 준 이미지 저장
FOOD_DB_PATH = "food_db.csv"        # 우리가 만든 영양 DB (name, serving_size, unit, calories, protein, fat, carbs)

# 텍스트 매칭용 파일들 (food_text_trainer.py --train 으로 생성)
VECTORIZER_PATH = "food_text_vectorizer.pkl"
FOOD_VECS_PATH = "food_db_tfidf.npz"
FOOD_META_PATH = "food_db_tfidf_meta.csv"

# 피드백에서 쌓이는 synonym 저장 파일
FEEDBACK_SYNONYMS_PATH = "feedback_synonyms.json"

DEVICE = torch.device("cpu")  # 필요하면 cuda 로 바꿀 수 있음

BATCH_SIZE = 16
FINETUNE_EPOCHS = 3
FINETUNE_LR = 1e-4

# 피드백 누적 학습 관련 설정
AUTO_FINETUNE_MIN_SAMPLES = 50  # 피드백 이미지가 이 이상 쌓이면 파인튜닝 추천

app = FastAPI()

# CORS: iOS 앱에서 호출할 수 있게 열어두기 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중엔 * 로, 실제 서비스 땐 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- 텍스트 유틸 / 매칭 관련 ----------------

def normalize_text(s: str) -> str:
    """검색/벡터라이저용 텍스트 정규화"""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("_", " ")
    # 영문자/숫자/공백만 남김
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


_text_matcher_loaded = False
_text_vectorizer = None
_food_vecs = None
_food_meta_df: Optional[pd.DataFrame] = None
_translator: Optional[Translator] = None


def load_text_matcher():
    """
    TF-IDF 벡터라이저와 food_db 메타/벡터를 로드.
    미리 food_text_trainer.py --train 을 한 번 돌려두어야 함.
    """
    global _text_matcher_loaded, _text_vectorizer, _food_vecs, _food_meta_df, _translator

    if _text_matcher_loaded:
        return

    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(FOOD_VECS_PATH) and os.path.exists(FOOD_META_PATH)):
        print("[WARN] 텍스트 매칭용 파일이 없어 text matcher 를 비활성화합니다.")
        _text_matcher_loaded = True
        return

    with open(VECTORIZER_PATH, "rb") as f:
        _text_vectorizer = pickle.load(f)

    _food_vecs = sparse.load_npz(FOOD_VECS_PATH)
    _food_meta_df = pd.read_csv(FOOD_META_PATH)

    if "name_norm" not in _food_meta_df.columns:
        _food_meta_df["name_norm"] = _food_meta_df["name"].astype(str).apply(normalize_text)

    if Translator is not None:
        _translator = Translator()
        print("[INFO] googletrans Translator 활성화됨.")
    else:
        _translator = None
        print("[WARN] googletrans 가 설치되지 않아 번역 기능이 비활성화됩니다.")

    _text_matcher_loaded = True
    print("[INFO] 텍스트 매칭 모델 로딩 완료")


def has_korean(text: str) -> bool:
    """문자열에 한글이 포함되었는지 간단히 검사"""
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def translate_to_en_if_korean(query: str) -> Tuple[str, bool]:
    """
    query에 한글이 있으면 googletrans로 영어로 번역 시도.
    - (번역된_텍스트, 번역했는지 여부) 반환
    - translator 없거나 실패하면 (원문, False) 반환
    """
    if not has_korean(query):
        return query, False

    if _translator is None:
        return query, False

    try:
        res = _translator.translate(query, src="ko", dest="en")
        return res.text, True
    except Exception as e:
        print(f"[WARN] 번역 실패: {e}")
        return query, False


STOP_WORDS = {
    "food", "foods", "dish", "dishes", "ingredient", "ingredients",
    "cuisine", "meal", "meals", "recipe", "recipes",
    "deep", "fried", "frying", "cooked", "cooking",
    "vegetable", "vegetables", "meat", "leaf", "leafy", "fast",
    "plate", "dining", "table", "lunch", "dinner"
}


def extract_keywords_from_labels(
    label_objs: List[Dict[str, Any]],
    max_keywords: int = 5
) -> List[str]:
    """
    Vision/텍스트 label 리스트에서 중요한 단어들을 추출.
    label_objs: [{"description": "...", "score": float}, ...]
    """
    keywords: List[str] = []
    if not label_objs:
        return keywords

    sorted_labels = sorted(label_objs, key=lambda x: x.get("score", 0.0), reverse=True)

    for obj in sorted_labels:
        desc = obj.get("description", "")
        norm = normalize_text(desc)
        for w in norm.split():
            if len(w) <= 2:
                continue
            if w in STOP_WORDS:
                continue
            if w not in keywords:
                keywords.append(w)
                if len(keywords) >= max_keywords:
                    return keywords
    return keywords


def search_food_candidates_masked(
    query: str,
    keywords: List[str],
    topk: int = 10
):
    """
    쿼리 문자열(영어 기준)로 TF-IDF 코사인 유사도를 사용해
    food_db에서 상위 topk 음식 후보를 반환.
    keywords가 주어지면 name_norm에 해당 키워드가 포함된 음식만 우선 후보로 사용.
    """
    load_text_matcher()
    if _text_vectorizer is None or _food_vecs is None or _food_meta_df is None:
        return []

    q_norm = normalize_text(query)
    if not q_norm:
        return []

    df = _food_meta_df
    if "name_norm" not in df.columns:
        df["name_norm"] = df["name"].astype(str).apply(normalize_text)

    # 키워드로 후보 필터링
    mask = None
    if keywords:
        series = df["name_norm"].fillna("")
        mask_arr = series.apply(lambda s: any(k in s for k in keywords)).to_numpy()
        if mask_arr.any():
            mask = mask_arr

    if mask is not None:
        df_cand = df[mask].copy()
        food_vecs_cand = _food_vecs[mask]
    else:
        df_cand = df
        food_vecs_cand = _food_vecs

    q_vec = _text_vectorizer.transform([q_norm])
    sims = cosine_similarity(q_vec, food_vecs_cand).flatten()

    topk = min(topk, len(sims))
    if topk <= 0:
        return []

    top_idx = np.argpartition(-sims, range(topk))[:topk]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    df_top = df_cand.iloc[top_idx].copy()
    df_top["match_score"] = sims[top_idx]

    candidates = []
    for row in df_top.itertuples():
        candidates.append({
            "name": row.name,
            "serving_size": float(row.serving_size),
            "unit": row.unit,
            "calories": float(row.calories),
            "protein": float(row.protein),
            "fat": float(row.fat),
            "carbs": float(row.carbs),
            "match_score": float(row.match_score),
        })
    return candidates

# ---------------- Google Vision API ----------------

_vision_client = None

def get_vision_client():
    global _vision_client
    if vision is None:
        return None
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


def get_vision_labels_from_bytes(
    image_bytes: bytes,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    raw 이미지 bytes로부터 Vision Label Detection 수행.
    """
    client = get_vision_client()
    if client is None:
        print("[WARN] google.cloud.vision 이 설치되지 않았거나, 사용할 수 없습니다.")
        return []

    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image, max_results=max_results)
    if response.error.message:
        print(f"[WARN] Vision API 에러: {response.error.message}")
        return []

    labels = []
    for l in response.label_annotations:
        labels.append({"description": l.description, "score": float(l.score)})
    return labels

# ---------------- Food-101 클래스 이름 처리 ----------------

def get_food101_class_names():
    ds = Food101(root="./data", split="train", download=False)
    return list(ds.classes)

def ensure_class_names():
    """class_names.txt가 이상하면 Food-101 기준으로 재생성."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        if len(names) == 101:
            print(f"[INFO] 기존 class_names.txt 사용 (클래스 수: {len(names)})")
            return names

        print(f"[WARN] class_names.txt 클래스 수 이상({len(names)}). 재생성 시도.")

    names = get_food101_class_names()
    print(f"[INFO] Food-101에서 클래스 이름 {len(names)}개 로드")

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")
    print("[INFO] class_names.txt 재생성 완료")
    return names

class_names: List[str] = ensure_class_names()
class_to_idx = {name: i for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)
class_names_norm = [normalize_text(n) for n in class_names]

# ---------------- synonym 로딩/저장 ----------------

_synonyms_cache: Optional[Dict[str, List[str]]] = None

def load_synonyms() -> Dict[str, List[str]]:
    global _synonyms_cache
    if _synonyms_cache is not None:
        return _synonyms_cache
    if os.path.exists(FEEDBACK_SYNONYMS_PATH):
        with open(FEEDBACK_SYNONYMS_PATH, "r", encoding="utf-8") as f:
            _synonyms_cache = json.load(f)
    else:
        _synonyms_cache = {}
    return _synonyms_cache

def save_synonyms(syn: Dict[str, List[str]]):
    global _synonyms_cache
    _synonyms_cache = syn
    with open(FEEDBACK_SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(syn, f, ensure_ascii=False, indent=2)

def add_synonyms_for_label(label: str, texts: List[str]):
    syn = load_synonyms()
    if label not in syn:
        syn[label] = []
    for t in texts:
        t = t.strip()
        if t and t not in syn[label]:
            syn[label].append(t)
    save_synonyms(syn)

# ---------------- 키워드 → Food-101 라벨 매핑 ----------------

def map_keywords_to_food101_label(
    keywords: List[str]
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    여러 키워드를 받아서, 그 중에서 Food-101 라벨과 가장 가까운 것을 찾는다.
    - keywords: 유저가 보낸 키워드 리스트 (ko/en 섞여 있음)
    - return: (mapped_label, keyword_en_map)
        mapped_label: Food-101 라벨명 (예: "fried_chicken") 또는 None
        keyword_en_map: {원래키워드: 영어/정규화된 키워드}
    """
    load_text_matcher()  # 번역기 세팅 위해

    keyword_en_map: Dict[str, str] = {}
    best_label = None
    best_score = 0

    for kw in keywords:
        raw = kw.strip()
        if not raw:
            continue

        # 한글이면 영어로 번역
        en, translated = translate_to_en_if_korean(raw)
        if translated:
            print(f"[INFO] 키워드 번역: '{raw}' -> '{en}'")
        keyword_en_map[raw] = en

        norm_en = normalize_text(en)
        if not norm_en:
            continue

        # 간단한 similarity: 토큰 overlap + 부분 문자열
        kw_tokens = set(norm_en.split())
        for cls_name, cls_norm in zip(class_names, class_names_norm):
            cls_tokens = set(cls_norm.split())

            overlap = len(kw_tokens & cls_tokens)
            score = overlap

            if cls_norm in norm_en or norm_en in cls_norm:
                score += 1

            if score > best_score:
                best_score = score
                best_label = cls_name

    if best_label:
        print(f"[INFO] 키워드 {keywords} -> Food-101 라벨 매핑 결과: {best_label} (score={best_score})")
    else:
        print(f"[WARN] 키워드 {keywords} 로 Food-101 라벨을 찾지 못했습니다.")

    return best_label, keyword_en_map

# ---------------- 모델 로드 ----------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"{MODEL_PATH} 가 없습니다. 먼저 학습 스크립트를 돌려주세요.")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes_ckpt = state_dict["classifier.1.weight"].shape[0]
    print(f"[INFO] 체크포인트 기준 클래스 수: {num_classes_ckpt}")

    if num_classes_ckpt != NUM_CLASSES:
        print(f"[WARN] checkpoint({num_classes_ckpt}) vs class_names({NUM_CLASSES}) 불일치.")

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes_ckpt)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- 공통 Transform ----------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- Custom Dataset (피드백 데이터) ----------------

class CustomLabeledDataset(Dataset):
    """
    custom_data/<label_name>/*.jpg 구조를 읽어서
    우리 class_to_idx 기준 label index로 반환.
    class_to_idx에 없는 라벨 폴더는 그냥 건너뜀.
    """
    def __init__(self, root, transform, class_to_idx):
        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        if not os.path.exists(root):
            return

        for label_name in os.listdir(root):
            label_dir = os.path.join(root, label_name)
            if not os.path.isdir(label_dir):
                continue
            if label_name not in class_to_idx:
                print(f"[WARN] custom_data에 '{label_name}' 폴더가 있지만 class_to_idx에 없음 -> 학습에서 제외")
                continue
            label_idx = class_to_idx[label_name]
            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    path = os.path.join(label_dir, fname)
                    self.samples.append((path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

# ---------------- 유틸 함수 ----------------

def pil_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_from_upload(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    return pil_from_bytes(contents)

def predict_tensor(tensor: torch.Tensor, topk: int = 3):
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=topk)
    results = []
    for p, idx in zip(top_probs, top_idxs):
        idx = idx.item()
        conf = float(p.item())
        raw_label = class_names[idx] if 0 <= idx < len(class_names) else f"unknown_{idx}"
        display_label = raw_label.replace("_", " ")
        results.append({"label": raw_label, "label_display": display_label, "confidence": conf})
    return results

def count_feedback_samples():
    """
    custom_data 폴더 내 이미지 개수와 라벨별 개수 집계
    """
    total = 0
    label_counts: Dict[str, int] = {}
    if not os.path.exists(CUSTOM_DATA_DIR):
        return {"total": 0, "labels": {}}

    for label_name in os.listdir(CUSTOM_DATA_DIR):
        label_dir = os.path.join(CUSTOM_DATA_DIR, label_name)
        if not os.path.isdir(label_dir):
            continue
        count = sum(
            1 for f in os.listdir(label_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        if count > 0:
            label_counts[label_name] = count
            total += count

    return {"total": total, "labels": label_counts}

# ---------------- food_db 유틸 ----------------

FOOD_DB_COLUMNS = ["name", "serving_size", "unit", "calories", "protein", "fat", "carbs"]

def load_food_db_df(limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    """
    food_db.csv를 메모리 안전하게 읽기 위한 함수.
    - limit, offset이 주어지면 전체가 아니라 앞에서 offset+limit 행까지만 읽고,
      그 중 offset~offset+limit 구간만 잘라서 반환한다.
    - limit가 None이면 전체를 읽지만, 이건 진짜 필요한 경우에만 사용.
    """
    if not os.path.exists(FOOD_DB_PATH):
        return pd.DataFrame(columns=FOOD_DB_COLUMNS)

    read_kwargs = {
        "usecols": lambda c: c in FOOD_DB_COLUMNS,  # 필요한 컬럼만 읽기
    }

    if limit is not None:
        read_kwargs["nrows"] = offset + limit if offset >= 0 else limit

    df = pd.read_csv(FOOD_DB_PATH, **read_kwargs)

    for col in FOOD_DB_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col not in ["name", "unit"] else ""

    if limit is not None:
        if offset < 0:
            offset = 0
        df = df.iloc[offset: offset + limit]

    return df[FOOD_DB_COLUMNS]


def save_food_db_df(df: pd.DataFrame):
    df.to_csv(FOOD_DB_PATH, index=False)

# ---------------- Pydantic 모델 ----------------

class FoodCreate(BaseModel):
    name: str
    carbs: float     # g
    protein: float   # g
    fat: float       # g
    serving_size: float = 1.0
    unit: str = "serving"
    calories: Optional[float] = None  # 없으면 계산 (4c,4p,9f)

class FoodSearchRequest(BaseModel):
    query: str
    topk: int = 10

# ---------------- API: 예측 ----------------

@app.post("/predict")
async def predict(image: UploadFile = File(...), topk: int = 3, food_topk: int = 5):
    """
    이미지 → (Food-101 + Google Vision) → 텍스트 매칭(TF-IDF) → food_db 후보 반환.

    - predictions: Food-101 라벨 및 confidence
    - vision_labels: Vision API 의 라벨 목록
    - query_text: 실제 텍스트 매칭에 사용된 쿼리
    - keywords: 라벨들에서 추출된 핵심 키워드
    - synonyms: top1 라벨에 대해 우리가 피드백으로 적립한 synonym 텍스트
    - candidates: food_db에서 찾은 음식 후보들
    """
    try:
        # 1) 이미지 bytes 한 번만 읽기
        image_bytes = await image.read()
        img = pil_from_bytes(image_bytes)

        # 2) Food-101 분류
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        predictions = predict_tensor(tensor, topk=topk)
        max_prob = predictions[0]["confidence"] if predictions else 0.0

        # 3) Vision 라벨
        vision_labels = get_vision_labels_from_bytes(image_bytes, max_results=10)

        # 4) 쿼리 텍스트/키워드 결정
        USE_FOOD101_THRESHOLD = 0.5

        if predictions and max_prob >= USE_FOOD101_THRESHOLD:
            print(f"[INFO] Food-101 신뢰도 {max_prob:.3f} >= {USE_FOOD101_THRESHOLD}, Food-101 기반 검색 사용")
            top1_label = predictions[0]["label_display"]
            query_text = top1_label
            fake_labels = [{"description": top1_label, "score": max_prob}]
            keywords = extract_keywords_from_labels(fake_labels, max_keywords=5)
        else:
            print(f"[INFO] Food-101 신뢰도 {max_prob:.3f} < {USE_FOOD101_THRESHOLD}, Vision 기반 검색 사용")
            query_text = " ".join(l["description"] for l in vision_labels)
            keywords = extract_keywords_from_labels(vision_labels, max_keywords=5)

        # 4-1) top1 Food-101 라벨의 synonym을 쿼리에 추가
        synonyms_for_top1: List[str] = []
        if predictions:
            top1_raw = predictions[0]["label"]  # 예: "fried_chicken"
            syn = load_synonyms().get(top1_raw, [])
            if syn:
                synonyms_for_top1 = syn
                query_text = query_text + " " + " ".join(syn)
                for s in syn:
                    norm = normalize_text(s)
                    if norm and norm not in keywords:
                        keywords.append(norm)

        # 5) TF-IDF 매칭
        candidates = search_food_candidates_masked(query_text, keywords, topk=food_topk)

        return JSONResponse({
            "predictions": predictions,
            "vision_labels": vision_labels,
            "query_text": query_text,
            "keywords": keywords,
            "synonyms": synonyms_for_top1,
            "candidates": candidates
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------- API: 피드백 (정답 라벨 + 키워드 저장) ----------------

@app.post("/feedback")
async def feedback(
    image: UploadFile = File(...),
    keywords: str = Form(...)
):
    """
    사용자가 '이 음식은 X야' 라고 알려줄 때 호출.
    - keywords: "치킨, 파닭, fried chicken" 처럼 여러 키워드 입력 가능.
    - 내부적으로:
        1) 각 키워드에 대해 ko->en 번역 시도
        2) Food-101 라벨과 가장 잘 맞는 canonical label을 찾음 (예: fried_chicken)
        3) 이미지는 custom_data/<canonical_label>/ 에 저장
        4) synonyms JSON에 canonical_label -> [영어 synonym들] 추가
    - 나중에 모델 헤드 학습 시, canonical_label만 사용됨.
    """
    # 1) 키워드 파싱
    raw_keywords = [k.strip() for k in keywords.replace(";", ",").split(",")]
    raw_keywords = [k for k in raw_keywords if k]

    if not raw_keywords:
        return JSONResponse({"error": "keywords 가 비어 있습니다. 예: '치킨, 파닭, fried chicken'"}, status_code=400)

    # 2) 키워드를 Food-101 라벨로 매핑 + 번역 맵 생성
    mapped_label, keyword_en_map = map_keywords_to_food101_label(raw_keywords)

    if not mapped_label:
        # 라벨은 못 찾았지만, 그래도 이미지를 어느 폴더에는 넣어두자 (디버깅용)
        os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
        unknown_dir = os.path.join(CUSTOM_DATA_DIR, "unmapped")
        os.makedirs(unknown_dir, exist_ok=True)

        existing = [
            f for f in os.listdir(unknown_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        fname = f"unmapped_{len(existing)}.jpg"
        save_path = os.path.join(unknown_dir, fname)

        img = pil_from_upload(image)
        img.save(save_path)

        stats = count_feedback_samples()
        can_finetune = stats["total"] >= AUTO_FINETUNE_MIN_SAMPLES

        return JSONResponse(
            {
                "message": "Food-101 라벨로 매핑하지 못해 unmapped 폴더에 저장했습니다.",
                "saved_path": save_path,
                "mapped_label": None,
                "keyword_translation_map": keyword_en_map,
                "feedback_stats": stats,
                "can_finetune": can_finetune,
                "finetune_threshold": AUTO_FINETUNE_MIN_SAMPLES,
            }
        )

    # 3) 이미지 저장 (canonical_label 기준)
    os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
    label_dir = os.path.join(CUSTOM_DATA_DIR, mapped_label)
    os.makedirs(label_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(label_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    fname = f"{mapped_label}_{len(existing)}.jpg"
    save_path = os.path.join(label_dir, fname)

    img = pil_from_upload(image)
    img.save(save_path)

    # 4) synonym(영어 기준) 적립
    english_synonyms = [en for en in keyword_en_map.values() if en.strip()]
    add_synonyms_for_label(mapped_label, english_synonyms)

    is_known = mapped_label in class_to_idx

    stats = count_feedback_samples()
    can_finetune = stats["total"] >= AUTO_FINETUNE_MIN_SAMPLES

    return JSONResponse(
        {
            "message": "피드백 데이터 저장 완료",
            "saved_path": save_path,
            "mapped_label": mapped_label,
            "is_known_class": is_known,
            "keyword_translation_map": keyword_en_map,
            "feedback_stats": stats,
            "can_finetune": can_finetune,
            "finetune_threshold": AUTO_FINETUNE_MIN_SAMPLES,
        }
    )

# ---------------- API: 피드백 통계 ----------------

@app.get("/feedback_stats")
async def feedback_stats():
    """
    custom_data/ 아래에 쌓인 피드백 이미지 개수를 확인.
    - total: 전체 이미지 수
    - labels: 라벨별 이미지 수
    - can_finetune: 파인튜닝 추천 여부
    """
    stats = count_feedback_samples()
    can_finetune = stats["total"] >= AUTO_FINETUNE_MIN_SAMPLES
    return JSONResponse({
        "total": stats["total"],
        "labels": stats["labels"],
        "can_finetune": can_finetune,
        "finetune_threshold": AUTO_FINETUNE_MIN_SAMPLES,
    })

# ---------------- API: 헤드 파인튜닝 ----------------

@app.post("/finetune_head")
async def finetune_head():
    """
    custom_data/ 에 쌓인 사용자 라벨 데이터를 가지고
    EfficientNet 헤드(classifier.1)만 몇 epoch 미세조정.
    - class_to_idx에 존재하는 라벨만 학습에 사용됨.
    """
    global model

    dataset = CustomLabeledDataset(CUSTOM_DATA_DIR, transform, class_to_idx)
    if len(dataset) == 0:
        return JSONResponse({"message": "custom_data 에 학습할 데이터가 없습니다."}, status_code=400)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=FINETUNE_LR)

    model.train()
    for epoch in range(FINETUNE_EPOCHS):
        running_loss = 0.0
        correct = total = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(dataloader)
        acc = correct / total if total > 0 else 0.0
        print(f"[FINETUNE] Epoch {epoch+1}/{FINETUNE_EPOCHS} "
              f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return JSONResponse(
        {"message": "헤드 파인튜닝 완료", "epochs": FINETUNE_EPOCHS}
    )

# ---------------- API: food_db 조회 ----------------

@app.get("/foods")
async def get_foods(limit: int = 100, offset: int = 0):
    """
    food_db에서 페이지네이션된 결과를 반환.
    - limit: 한 번에 몇 개 가져올지 (기본 100)
    - offset: 어디서부터 가져올지 (기본 0)
    """
    if limit <= 0:
        limit = 50

    df = load_food_db_df(limit=limit, offset=offset)
    foods = df.to_dict(orient="records")

    return JSONResponse({
        "foods": foods,
        "limit": limit,
        "offset": offset,
        "count": len(foods),
    })


@app.get("/foods/names")
async def get_food_names(limit: int = 200, offset: int = 0):
    """
    food_db에 저장된 음식 이름만 부분 리스트로 반환.
    - 너무 많은 이름을 한 번에 보내면 또 무거워지니까 기본 200개 정도만.
    """
    df = load_food_db_df(limit=limit, offset=offset)
    names = df["name"].dropna().astype(str).tolist()
    return JSONResponse({
        "names": names,
        "limit": limit,
        "offset": offset,
        "count": len(names),
    })

# ---------------- API: 텍스트로 음식 검색 (한국어 → 영어 번역 포함) ----------------

@app.post("/foods/search")
async def search_foods(req: FoodSearchRequest):
    """
    사용자가 텍스트로 입력한 음식 이름(한글/영어)을 기반으로
    food_db에서 비슷한 음식들 상위 topk를 반환.

    - query가 한국어면 영어로 번역해서 검색
    - TF-IDF 코사인 유사도 기반
    """
    q_raw = req.query.strip()
    if not q_raw:
        return JSONResponse({"error": "query is empty"}, status_code=400)

    load_text_matcher()
    q_en, translated = translate_to_en_if_korean(q_raw)
    candidates = search_food_candidates_masked(q_en, keywords=[], topk=req.topk)

    return JSONResponse({
        "query": q_raw,
        "used_query": q_en,
        "translated": translated,
        "candidates": candidates,
    })

# ---------------- API: 새 음식 추가 ----------------

@app.post("/foods/add")
async def add_food(food: FoodCreate):
    """
    새 음식 + 탄/단/지(필수) + 선택적인 칼로리, 서빙 단위 등을
    food_db.csv에 추가한다.

    - 동일 이름(대소문자 무시)이 이미 있으면 에러 반환.
    - calories가 비어 있으면 4*carbs + 4*protein + 9*fat 로 계산.
    """
    df = load_food_db_df()

    name_lower = food.name.strip().lower()
    if not df.empty:
        exists = df["name"].astype(str).str.lower() == name_lower
        if exists.any():
            return JSONResponse(
                {"error": f"이미 존재하는 음식입니다: {food.name}"},
                status_code=400
            )

    calories = food.calories
    if calories is None:
        calories = 4.0 * food.carbs + 4.0 * food.protein + 9.0 * food.fat

    new_row = {
        "name": food.name.strip(),
        "serving_size": float(food.serving_size),
        "unit": food.unit,
        "calories": float(calories),
        "protein": float(food.protein),
        "fat": float(food.fat),
        "carbs": float(food.carbs),
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_food_db_df(df)

    return JSONResponse(
        {
            "message": "새 음식이 food_db에 추가되었습니다.",
            "food": new_row,
        }
    )

# ---------------- main: python server.py 로 바로 실행 가능 ----------------

if __name__ == "__main__":
    import uvicorn
    print("[INFO] 서버 시작: http://0.0.0.0:8000  (docs: /docs)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
