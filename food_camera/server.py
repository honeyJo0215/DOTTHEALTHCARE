# server.py
import os
import io
import json
import re
import requests   # ⬅ 새로 추가
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

# 번역 / Vision (있으면 사용, 없으면 자동 비활성)
try:
    from googletrans import Translator
except ImportError:
    Translator = None

try:
    from google.cloud import vision
except ImportError:
    vision = None

# ---------------- 기본 설정 ----------------
# ---- 온라인 이미지 검색 (Bing Image Search API 사용 예시) ----
BING_IMAGE_SEARCH_KEY = os.environ.get("BING_IMAGE_SEARCH_KEY")  # Azure에서 발급받은 키
BING_IMAGE_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"

if not BING_IMAGE_SEARCH_KEY:
    print("[WARN] BING_IMAGE_SEARCH_KEY 가 설정되어 있지 않습니다. auto_collect_subdata 는 실제 이미지 다운로드를 하지 않습니다.")


MODEL_PATH = "food_classifier.pth"
CLASS_NAMES_PATH = "class_names.txt"
CUSTOM_DATA_DIR = "./custom_data"   # 부모 라벨용 피드백 이미지
FOOD_DB_PATH = "food_db.csv"        # name, serving_size, unit, calories, protein, fat, carbs

# 텍스트 매칭용 (food_text_trainer.py --train 으로 생성)
VECTORIZER_PATH = "food_text_vectorizer.pkl"
FOOD_VECS_PATH = "food_db_tfidf.npz"
FOOD_META_PATH = "food_db_tfidf_meta.csv"

# 피드백에서 쌓이는 synonym 저장 파일
FEEDBACK_SYNONYMS_PATH = "feedback_synonyms.json"

# ▶ 새로운: 부모별 서브클래스 관리
SUB_DATA_DIR = "./sub_data"               # sub_data/<parent>/<child>/*.jpg
SUBCLASS_META_PATH = "subclass_meta.json" # 부모별 children 목록
SUBHEAD_DIR = "./subheads"               # 부모별 서브헤드 weight 저장

DEVICE = torch.device("cpu")  # 필요하면 cuda 로 변경 가능

BATCH_SIZE = 16
FINETUNE_EPOCHS = 3
FINETUNE_LR = 1e-4

# 서브헤드 학습용
SUBHEAD_EPOCHS = 5
SUBHEAD_LR = 1e-3

AUTO_FINETUNE_MIN_SAMPLES = 50  # 피드백 이미지가 이 이상 쌓이면 파인튜닝 추천

app = FastAPI()

# CORS: iOS 앱에서 호출할 수 있게 열어두기 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 서비스 시 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- 텍스트 유틸 / 번역 / 매칭 ----------------

def normalize_text(s: str) -> str:
    """검색/벡터라이저용 텍스트 정규화 (소문자 + 영문/숫자/공백만)"""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("_", " ")
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


_text_matcher_loaded = False
_text_vectorizer = None
_food_vecs = None
_food_meta_df: Optional[pd.DataFrame] = None
_translator: Optional[Translator] = None


def ensure_translator():
    """googletrans Translator 초기화 (있을 때만)"""
    global _translator
    if _translator is None and Translator is not None:
        try:
            _translator = Translator()
            print("[INFO] googletrans Translator 초기화 완료.")
        except Exception as e:
            print(f"[WARN] Translator 초기화 실패: {e}")
            _translator = None


_text_matcher_loaded = False
_text_vectorizer = None
_food_vecs = None
_food_meta_df: Optional[pd.DataFrame] = None

def rebuild_text_index_from_food_db():
    """
    현재 food_db.csv 내용을 기반으로
    _food_meta_df, _food_vecs 를 다시 만든다.
    - trainer 는 vectorizer.pkl 만 만들면 되고,
      tf-idf 인덱스는 항상 최신 food_db 로부터 계산된다.
    """
    global _food_meta_df, _food_vecs, _text_vectorizer

    if _text_vectorizer is None:
        return

    df = load_food_db_df(limit=None, offset=0)
    if df.empty:
        _food_meta_df = None
        _food_vecs = None
        return

    df = df.copy()
    # 정규화된 이름 컬럼
    df["name_norm"] = df["name"].astype(str).apply(normalize_text)
    texts = df["name_norm"].fillna("").tolist()

    # 기존 vocab/idf 를 사용해서 현재 DB 전체에 대해 TF-IDF 벡터 생성
    _food_vecs = _text_vectorizer.transform(texts)
    _food_meta_df = df
    print(f"[INFO] 텍스트 인덱스 재빌드 완료: {len(df)} foods")


def load_text_matcher():
    """
    TF-IDF 벡터라이저를 로드하고,
    현재 food_db.csv를 기준으로 텍스트 인덱스를 만든다.
    (이제 더 이상 FOOD_VECS_PATH, FOOD_META_PATH 는 필수가 아님)
    """
    global _text_matcher_loaded, _text_vectorizer

    if _text_matcher_loaded:
        return

    if not os.path.exists(VECTORIZER_PATH):
        print("[WARN] VECTORIZER_PATH 가 없어 텍스트 매칭을 비활성화합니다.")
        _text_matcher_loaded = True
        return

    with open(VECTORIZER_PATH, "rb") as f:
        _text_vectorizer = pickle.load(f)

    ensure_translator()
    # 🔥 여기서 바로 현재 food_db 기준으로 인덱스를 만든다
    rebuild_text_index_from_food_db()

    _text_matcher_loaded = True
    print("[INFO] 텍스트 매칭 모델 로딩 + 인덱스 생성 완료")


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

    ensure_translator()
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
        print("[WARN] google.cloud.vision 사용 불가 (미설치 또는 인증 문제).")
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

# ---------------- 서브클래스 메타/헤드 관리 ----------------

_subclass_meta: Optional[Dict[str, Any]] = None
_subhead_cache: Dict[str, nn.Module] = {}


def load_subclass_meta() -> Dict[str, Any]:
    global _subclass_meta
    if _subclass_meta is not None:
        return _subclass_meta
    if os.path.exists(SUBCLASS_META_PATH):
        with open(SUBCLASS_META_PATH, "r", encoding="utf-8") as f:
            _subclass_meta = json.load(f)
    else:
        _subclass_meta = {}
    return _subclass_meta


def save_subclass_meta(meta: Dict[str, Any]):
    global _subclass_meta
    _subclass_meta = meta
    with open(SUBCLASS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def register_subclass_label(parent_label: str, child_id: str, child_display: str):
    """
    parent_label 아래에 child_id(예: green_onion_chicken)를 서브클래스로 등록.
    - parent 자체도 children 리스트에 항상 포함되도록 보장.
    """
    meta = load_subclass_meta()
    if parent_label not in meta:
        meta[parent_label] = {"children": []}

    children = meta[parent_label]["children"]

    # 1) 부모 라벨을 children에 넣어두기
    parent_exists = any(c["id"] == parent_label for c in children)
    if not parent_exists:
        children.append({
            "id": parent_label,
            "display": parent_label.replace("_", " ")
        })

    # 2) 새 child 추가
    exists = any(c["id"] == child_id for c in children)
    if not exists:
        children.append({"id": child_id, "display": child_display})

    meta[parent_label]["children"] = children
    save_subclass_meta(meta)
    print(f"[INFO] 서브클래스 등록: parent={parent_label}, child={child_id} ({child_display})")


def extract_subclass_id(en_text: str) -> Tuple[str, str]:
    """
    영어 텍스트(en_text)로부터 서브클래스 id와 display 생성.
    - id: green_onion_chicken
    - display: green onion chicken
    """
    norm = normalize_text(en_text)  # "green onion chicken"
    if not norm:
        return "custom_food", en_text.strip() or "custom food"
    sub_id = norm.replace(" ", "_")  # "green_onion_chicken"
    display = norm  # 그대로 display 사용
    return sub_id, display


def load_subhead(parent_label: str) -> Optional[nn.Module]:
    """
    parent_label에 대한 서브헤드 Linear(in=1280, out=n_children) 로드.
    없으면 None.
    """
    if parent_label in _subhead_cache:
        return _subhead_cache[parent_label]

    meta = load_subclass_meta()
    if parent_label not in meta:
        return None

    children = meta[parent_label].get("children", [])
    if len(children) < 2:
        return None

    head_path = os.path.join(SUBHEAD_DIR, f"{parent_label}_head.pth")
    if not os.path.exists(head_path):
        return None

    # head 크기는 children 수에 따라 결정
    out_dim = len(children)
    head = nn.Linear(1280, out_dim)
    state_dict = torch.load(head_path, map_location=DEVICE)
    head.load_state_dict(state_dict)
    head.to(DEVICE)
    head.eval()

    _subhead_cache[parent_label] = head
    print(f"[INFO] 서브헤드 로드 완료: {parent_label}, subclasses={out_dim}")
    return head

# ---------------- 키워드 → Food-101 부모 라벨 매핑 ----------------

def map_keywords_to_food101_label(
    keywords: List[str]
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    여러 키워드를 받아서, 그 중에서 Food-101 라벨과 가장 가까운 것을 찾는다.
    - 반환: (부모 Food-101 라벨, {원래키워드: 영어번역텍스트})
    """
    load_text_matcher()  # 번역기 세팅 위해

    keyword_en_map: Dict[str, str] = {}
    best_label = None
    best_score = 0

    for kw in keywords:
        raw = kw.strip()
        if not raw:
            continue

        en, translated = translate_to_en_if_korean(raw)
        if translated:
            print(f"[INFO] 키워드 번역: '{raw}' -> '{en}'")
        keyword_en_map[raw] = en

        norm_en = normalize_text(en)
        if not norm_en:
            continue

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

# ---------------- 모델 로드 및 feature 추출 ----------------

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


def extract_features(x: torch.Tensor) -> torch.Tensor:
    """
    EfficientNet 중간 feature 추출 (classifier 앞 1280차원 벡터)
    - 항상 eval 모드에서 dropout/bn 고정
    """
    model.eval()
    with torch.no_grad():
        feats = model.features(x)
        feats = model.avgpool(feats)
        feats = torch.flatten(feats, 1)
    return feats


# ---------------- 공통 Transform ----------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- Custom Dataset (부모 라벨용) ----------------

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

# ---------------- 서브클래스용 Dataset ----------------

class SubclassDataset(Dataset):
    """
    sub_data/<parent>/<child>/*.jpg 구조를 읽어서
    parent_label에 대한 child index(0..n_child-1)를 반환.
    """
    def __init__(self, parent_label: str, child_ids: List[str], transform):
        self.samples = []
        self.transform = transform
        self.child_ids = child_ids
        root = os.path.join(SUB_DATA_DIR, parent_label)
        if not os.path.exists(root):
            return

        for ci, cid in enumerate(child_ids):
            child_dir = os.path.join(root, cid)
            if not os.path.isdir(child_dir):
                continue
            for fname in os.listdir(child_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                path = os.path.join(child_dir, fname)
                self.samples.append((path, ci))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ci = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, ci

# ---------------- 유틸 함수 ----------------

def slugify_label(name: str) -> str:
    """
    음식 이름을 파일/폴더 이름으로 쓸 수 있는 슬러그로 변환.
    - 영문/숫자만 남기고 나머지는 '_' 로 치환.
    - 전부 날아가면 'food' 로 대체.
    """
    s = name.strip().lower()
    # 한글 등 비ASCII는 일단 '_' 로 처리 (필요하면 translator로 영어 변환해서 쓰면 됨)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "food"
    return s

def search_food_images_online(query: str, count: int = 3) -> List[str]:
    """
    Bing Image Search API로 음식 이미지 URL들을 가져온다.
    - query: 검색어 (예: 'fried chicken food')
    - count: 가져올 최대 이미지 수
    """
    if not BING_IMAGE_SEARCH_KEY:
        print("[WARN] BING_IMAGE_SEARCH_KEY 가 설정되어 있지 않습니다.")
        return []

    headers = {
        "Ocp-Apim-Subscription-Key": BING_IMAGE_SEARCH_KEY
    }
    params = {
        "q": query,
        "count": count,
        "safeSearch": "Strict",
        "imageType": "Photo",
    }

    try:
        resp = requests.get(BING_IMAGE_SEARCH_ENDPOINT, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        urls = [item["contentUrl"] for item in data.get("value", []) if "contentUrl" in item]
        return urls
    except Exception as e:
        print(f"[WARN] 이미지 검색 실패: query='{query}', error={e}")
        return []
    
def download_image_from_url(url: str) -> Optional[Image.Image]:
    """
    주어진 URL에서 이미지를 다운로드하여 PIL Image로 반환.
    실패 시 None.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] 이미지 다운로드 실패: url='{url}', error={e}")
        return None

def predict_parent_label_from_pil(img: Image.Image) -> Optional[str]:
    """
    PIL 이미지를 EfficientNet(Food-101 헤드)로 추론하여
    top-1 부모 라벨을 반환.
    """
    model.eval()
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    idx = top_idx.item()
    if idx < 0 or idx >= len(class_names):
        return None
    return class_names[idx]

def save_sub_image(img: Image.Image, parent_label: str, child_id: str) -> str:
    """
    서브 데이터 이미지 한 장을
    sub_data/<parent_label>/<child_id>/<child_id>_N.jpg 로 저장하고 경로를 반환.
    """
    parent_dir = os.path.join(SUB_DATA_DIR, parent_label)
    child_dir = os.path.join(parent_dir, child_id)
    os.makedirs(child_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(child_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    fname = f"{child_id}_{len(existing)}.jpg"
    path = os.path.join(child_dir, fname)
    img.save(path)
    return path


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


def list_finetune_items() -> List[Dict[str, Any]]:
    """
    custom_data/ 아래의 모든 이미지를 평탄화해서
    [{"id": 0, "label": "...", "filename": "...", "path": "..."}] 형태로 반환.
    """
    items: List[Dict[str, Any]] = []
    idx = 0
    if os.path.exists(CUSTOM_DATA_DIR):
        for label in sorted(os.listdir(CUSTOM_DATA_DIR)):
            label_dir = os.path.join(CUSTOM_DATA_DIR, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                path = os.path.join(label_dir, fname)
                items.append({
                    "id": idx,
                    "label": label,
                    "filename": fname,
                    "path": path,
                })
                idx += 1
    return items

# ---------------- food_db 유틸 ----------------

FOOD_DB_COLUMNS = ["name", "serving_size", "unit", "calories", "protein", "fat", "carbs"]


def load_food_db_df(limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    """
    food_db.csv를 메모리 안전하게 읽기 위한 함수.
    """
    if not os.path.exists(FOOD_DB_PATH):
        return pd.DataFrame(columns=FOOD_DB_COLUMNS)

    read_kwargs = {
        "usecols": lambda c: c in FOOD_DB_COLUMNS,
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


def prepare_db_name(src_name: str) -> Tuple[str, Optional[str], bool]:
    """
    DB에 저장할 이름을 준비:
    - 한글이면 영어로 번역해서 영어를 name으로 사용
    - (stored_name, original_name_if_translated_else_None, translated_flag)
    """
    src_name = src_name.strip()
    if not src_name:
        return src_name, None, False

    en, translated = translate_to_en_if_korean(src_name)
    stored = en.strip() if en.strip() else src_name
    return stored, (src_name if translated else None), translated

# ---------------- Pydantic 모델 ----------------

class AutoCollectRequest(BaseModel):
    start_index: int = 0       # food_db에서 시작할 인덱스
    max_items: int = 10        # 이번 호출에서 처리할 음식 개수
    images_per_food: int = 3   # 음식 하나당 다운로드할 이미지 수
    language: str = "en"       # 검색용 언어 (지금은 'en'만 사용)


class FoodCreate(BaseModel):
    name: str
    carbs: float     # g
    protein: float   # g
    fat: float       # g
    serving_size: float = 1.0
    unit: str = "serving"
    calories: Optional[float] = None


class FoodSearchRequest(BaseModel):
    query: str
    topk: int = 10

# ---------------- DB에 음식 추가하는 내부 함수 ----------------

def add_food_row(
    name: str,
    carbs: float,
    protein: float,
    fat: float,
    serving_size: float = 1.0,
    unit: str = "serving",
    calories: Optional[float] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    /foods/add와 /feedback에서 공통으로 사용하는 내부 DB 추가 함수.
    - name에 한글이 있으면 영어로 번역해서 저장.
    """
    df = load_food_db_df()

    stored_name, original_name, translated = prepare_db_name(name)

    name_lower = stored_name.lower()
    if not df.empty:
        exists = df["name"].astype(str).str.lower() == name_lower
        if exists.any():
            raise ValueError(f"이미 존재하는 음식입니다: {stored_name}")

    if calories is None:
        calories = 4.0 * carbs + 4.0 * protein + 9.0 * fat

    new_row = {
        "name": stored_name,
        "serving_size": float(serving_size),
        "unit": unit,
        "calories": float(calories),
        "protein": float(protein),
        "fat": float(fat),
        "carbs": float(carbs),
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_food_db_df(df)

    meta = {
        "stored_name": stored_name,
        "original_name": original_name or name,
        "translated": translated,
    }
    return new_row, meta

# ---------------- 서브헤드 inference ----------------

def apply_subclass_head(tensor: torch.Tensor, base_pred: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    base_pred 에서 부모 라벨을 꺼내고, 해당 parent에 대한 서브헤드가 있으면
    1280차원 feature를 넣어 서브라벨을 예측.
    """
    parent_label = base_pred["label"]
    meta = load_subclass_meta()
    if parent_label not in meta:
        return None

    children = meta[parent_label].get("children", [])
    if len(children) < 2:
        return None

    head = load_subhead(parent_label)
    if head is None:
        return None

    head.eval()
    with torch.no_grad():
        feats = extract_features(tensor.to(DEVICE))
        logits = head(feats)
        probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    ci = top_idx.item()
    child = children[ci]
    sub_id = child["id"]
    sub_display = child.get("display", sub_id.replace("_", " "))

    return {
        "parent_label": parent_label,
        "sub_label": sub_id,
        "sub_label_display": sub_display,
        "confidence": float(top_prob.item()),
    }

def apply_best_subhead_any_parent(tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
    """
    부모 예측이 틀렸을 경우를 대비한 fallback:
    - subclass_meta에 등록된 모든 parent에 대해 서브헤드를 돌려보고
    - 전체 parent/child 중 가장 확률이 높은 조합을 하나 반환.
    """
    meta = load_subclass_meta()
    if not meta:
        return None

    # 공통 feature 추출 (1번만)
    feats = extract_features(tensor.to(DEVICE))

    best = None

    for parent_label, info in meta.items():
        children = info.get("children", [])
        if len(children) < 2:
            continue

        head = load_subhead(parent_label)
        if head is None:
            continue

        head.eval()
        with torch.no_grad():
            logits = head(feats)  # [1, n_child]
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

        ci = top_idx.item()
        child = children[ci]
        sub_id = child["id"]
        sub_display = child.get("display", sub_id.replace("_", " "))

        conf = float(top_prob.item())

        if (best is None) or (conf > best["confidence"]):
            best = {
                "parent_label": parent_label,
                "sub_label": sub_id,
                "sub_label_display": sub_display,
                "confidence": conf,
                "from_fallback": True,
            }

    return best

# ---------------- API: 예측 ----------------

@app.post("/predict")
async def predict(image: UploadFile = File(...), topk: int = 3, food_topk: int = 5):
    """
    이미지 → (EfficientNet 부모 + (있으면) 서브헤드) → 텍스트 매칭(TF-IDF) → food_db 후보 반환.
    """
    try:
        image_bytes = await image.read()
        img = pil_from_bytes(image_bytes)

        tensor = transform(img).unsqueeze(0).to(DEVICE)
        predictions = predict_tensor(tensor, topk=topk)
        max_prob = predictions[0]["confidence"] if predictions else 0.0

        # 서브헤드가 있다면 부모 라벨에 대해 서브라벨 보정
        sub_prediction = None
        if predictions:
            sub_prediction = apply_subclass_head(tensor, predictions[0])

        # Vision 라벨 (텍스트 보조)
        vision_labels = get_vision_labels_from_bytes(image_bytes, max_results=10)
        # 2차 시도(폴백): 부모 예측이 틀렸거나 서브헤드가 없는 경우,
        # 모든 parent의 서브헤드를 돌려서 한 번이라도 best subclass를 찾아본다.
        if sub_prediction is None:
            sub_prediction = apply_best_subhead_any_parent(tensor)
        USE_FOOD101_THRESHOLD = 0.5
        SUBLABEL_THRESHOLD = 0.4

        # 텍스트 매칭에 사용할 라벨 텍스트 결정
        if sub_prediction and sub_prediction["confidence"] >= SUBLABEL_THRESHOLD:
            # 서브라벨이 충분히 자신있을 때 서브라벨 텍스트 사용
            top_label_text = sub_prediction["sub_label_display"]
        else:
            # 서브헤드 없거나 신뢰도 낮으면 부모 라벨 사용
            if predictions and max_prob >= USE_FOOD101_THRESHOLD:
                top_label_text = predictions[0]["label_display"]
            else:
                top_label_text = None

        if top_label_text is not None:
            print(f"[INFO] 텍스트 매칭에 사용할 라벨 텍스트: {top_label_text}")
            query_text = top_label_text
            fake_labels = [{"description": top_label_text, "score": max_prob}]
            keywords = extract_keywords_from_labels(fake_labels, max_keywords=5)
        else:
            # Food-101 신뢰도가 낮으면 Vision 라벨들의 텍스트를 사용
            query_text = " ".join(l["description"] for l in vision_labels)
            keywords = extract_keywords_from_labels(vision_labels, max_keywords=5)
            # Vision 라벨도 없으면, 어쩔 수 없이 top-1 Food-101 라벨이라도 사용
            if not query_text and predictions:
                query_text = predictions[0]["label_display"]
        # 부모 라벨에 걸린 synonym 도 쿼리에 추가
        synonyms_for_parent: List[str] = []
        if predictions:
            top1_raw = predictions[0]["label"]
            syn = load_synonyms().get(top1_raw, [])
            if syn:
                synonyms_for_parent = syn
                query_text = query_text + " " + " ".join(syn)
                for s in syn:
                    norm = normalize_text(s)
                    if norm and norm not in keywords:
                        keywords.append(norm)

        candidates = search_food_candidates_masked(query_text, keywords, topk=food_topk)

        return JSONResponse({
            "predictions": predictions,
            "sub_prediction": sub_prediction,
            "vision_labels": vision_labels,
            "query_text": query_text,
            "keywords": keywords,
            "synonyms": synonyms_for_parent,
            "candidates": candidates
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------- API: 피드백 (키워드 + 서브라벨 + DB 추가) ----------------

@app.post("/feedback")
async def feedback(
    image: UploadFile = File(...),
    keywords: str = Form(...),         # "치킨, 파닭, fried chicken"
    main_keyword: Optional[str] = Form(None),  # 이 중 서브라벨로 쓸 하나 (없으면 첫 번째)
    use_for_training: bool = Form(True),
    add_to_db: bool = Form(False),
    db_name: Optional[str] = Form(None),
    db_carbs: Optional[float] = Form(None),
    db_protein: Optional[float] = Form(None),
    db_fat: Optional[float] = Form(None),
    db_serving_size: float = Form(1.0),
    db_unit: str = Form("serving"),
    db_calories: Optional[float] = Form(None),
):
    """
    사용자가 '이 음식은 X야' 라고 알려줄 때 호출.

    - keywords: "치킨, 파닭, fried chicken" 처럼 여러 키워드 입력 가능.
    - main_keyword: 그 중 서브라벨로 사용할 하나 (예: "파닭")
    - use_for_training:
        True  -> 부모 Food-101 라벨 매핑 + custom_data + sub_data에 이미지 저장 + synonym 적립
        False -> 학습에 사용하지 않음 (선택적으로 add_to_db만 수행)
    - add_to_db:
        True 이고 macros가 주어지면 food_db에 새 음식으로 추가
    """
    # 0) 키워드 파싱
    raw_keywords = [k.strip() for k in keywords.replace(";", ",").split(",")]
    raw_keywords = [k for k in raw_keywords if k]

    if not raw_keywords:
        return JSONResponse({"error": "keywords 가 비어 있습니다. 예: '치킨, 파닭, fried chicken'"}, status_code=400)

    if main_keyword is None or main_keyword.strip() not in raw_keywords:
        main_keyword = raw_keywords[0]
    main_keyword = main_keyword.strip()

    # 1) DB 추가 (옵션)
    db_result = None
    db_error = None
    if add_to_db:
        if db_name is None or db_carbs is None or db_protein is None or db_fat is None:
            db_error = "add_to_db=True 인 경우 db_name, db_carbs, db_protein, db_fat 이 모두 필요합니다."
        else:
            try:
                new_row, meta = add_food_row(
                    name=db_name,
                    carbs=db_carbs,
                    protein=db_protein,
                    fat=db_fat,
                    serving_size=db_serving_size,
                    unit=db_unit,
                    calories=db_calories,
                )
                db_result = {
                    "food": new_row,
                    "meta": meta,
                }
            except ValueError as ve:
                db_error = str(ve)
            except Exception as e:
                db_error = f"DB 추가 중 오류: {e}"

    # 2) 학습용 피드백 (부모 + 서브라벨)
    mapped_label = None
    keyword_en_map: Dict[str, str] = {}
    saved_path = None
    is_known = False
    sub_label_id = None
    sub_label_display = None

    if use_for_training:
        mapped_label, keyword_en_map = map_keywords_to_food101_label(raw_keywords)

        img = pil_from_upload(image)

        if not mapped_label:
            # 부모 라벨 못찾으면 unmapped 폴더에만 저장
            os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
            unknown_dir = os.path.join(CUSTOM_DATA_DIR, "unmapped")
            os.makedirs(unknown_dir, exist_ok=True)

            existing = [
                f for f in os.listdir(unknown_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            fname = f"unmapped_{len(existing)}.jpg"
            saved_path = os.path.join(unknown_dir, fname)
            img.save(saved_path)

            stats = count_feedback_samples()
            can_finetune = stats["total"] >= AUTO_FINETUNE_MIN_SAMPLES

            return JSONResponse(
                {
                    "message": "Food-101 부모 라벨로 매핑하지 못해 unmapped 폴더에 저장했습니다.",
                    "saved_path": saved_path,
                    "mapped_label": None,
                    "is_known_class": False,
                    "keyword_translation_map": keyword_en_map,
                    "feedback_stats": stats,
                    "can_finetune": can_finetune,
                    "finetune_threshold": AUTO_FINETUNE_MIN_SAMPLES,
                    "db_added": db_result,
                    "db_error": db_error,
                }
            )

        # 2-1) 부모 라벨 기준 custom_data 저장 (기존 parent head finetune용)
        os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
        label_dir = os.path.join(CUSTOM_DATA_DIR, mapped_label)
        os.makedirs(label_dir, exist_ok=True)

        existing = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        fname = f"{mapped_label}_{len(existing)}.jpg"
        saved_path = os.path.join(label_dir, fname)
        img.save(saved_path)

        # 2-2) synonym 적립 (부모 라벨 기준)
        english_synonyms = [en for en in keyword_en_map.values() if en.strip()]
        add_synonyms_for_label(mapped_label, english_synonyms)

        is_known = mapped_label in class_to_idx

        # 2-3) 서브라벨 (main_keyword) 기준으로 sub_data 저장 + subclass_meta 업데이트
        main_en = keyword_en_map.get(main_keyword, main_keyword)
        sub_id, sub_disp = extract_subclass_id(main_en)

        register_subclass_label(mapped_label, sub_id, sub_disp)

        os.makedirs(SUB_DATA_DIR, exist_ok=True)
        parent_dir = os.path.join(SUB_DATA_DIR, mapped_label)
        child_dir = os.path.join(parent_dir, sub_id)
        os.makedirs(child_dir, exist_ok=True)

        existing_sub = [
            f for f in os.listdir(child_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        sub_fname = f"{sub_id}_{len(existing_sub)}.jpg"
        sub_save_path = os.path.join(child_dir, sub_fname)
        img.save(sub_save_path)

        sub_label_id = sub_id
        sub_label_display = sub_disp

    stats = count_feedback_samples()
    can_finetune = stats["total"] >= AUTO_FINETUNE_MIN_SAMPLES

    return JSONResponse(
        {
            "message": "피드백 처리 완료",
            "saved_path": saved_path,
            "mapped_label": mapped_label,          # 부모 Food-101 라벨
            "sub_label_id": sub_label_id,          # 서브라벨 id (예: green_onion_chicken)
            "sub_label_display": sub_label_display,# 서브라벨 표시용 텍스트
            "is_known_class": is_known,
            "keyword_translation_map": keyword_en_map,
            "feedback_stats": stats,
            "can_finetune": can_finetune,
            "finetune_threshold": AUTO_FINETUNE_MIN_SAMPLES,
            "db_added": db_result,
            "db_error": db_error,
        }
    )

@app.post("/auto_collect_subdata_not_used_yet")
async def auto_collect_subdata(req: AutoCollectRequest):
    """
    food_db에 있는 음식 이름들을 자동으로 순회하면서:
    1) 음식 이름으로 인터넷에서 이미지 검색
    2) 이미지를 다운로드
    3) EfficientNet(Food-101)로 부모 라벨 예측
    4) parent_label 밑에 child_id(= food_db 이름 기반) 서브라벨로 등록
    5) 이미지를 sub_data/<parent>/<child>/ 에 저장

    - 한 번 호출할 때 max_items 개까지만 처리
    - 클라이언트가 start_index를 바꿔가며 여러 번 호출하면,
      사실상 food_db 전체를 순차적으로 처리할 수 있음.
    """
    # 0) food_db 전체 로드 (용량이 너무 크면 나중에 chunksize로 바꿀 수 있음)
    df = load_food_db_df(limit=None)
    total = len(df)
    if total == 0:
        return JSONResponse(
            {"message": "food_db에 데이터가 없습니다.", "processed": 0},
            status_code=400,
        )

    start = max(0, req.start_index)
    end = min(total, start + req.max_items)
    if start >= total:
        return JSONResponse(
            {
                "message": "start_index가 food_db 범위를 벗어났습니다.",
                "total": total,
                "processed": 0,
            },
            status_code=400,
        )

    processed_items = []

    for idx in range(start, end):
        row = df.iloc[idx]
        food_name = str(row["name"]).strip()
        if not food_name:
            continue

        # 검색용 쿼리 구성 (간단히 음식 이름 + food)
        query = f"{food_name} food"
        print(f"[AUTO] ({idx}) '{food_name}' -> query='{query}'")

        urls = search_food_images_online(query, count=req.images_per_food)
        if not urls:
            print(f"[AUTO] 이미지 검색 실패: {food_name}")
            continue

        child_display = food_name
        child_id = slugify_label(food_name)

        parent_labels_used = set()
        saved_count = 0

        for url in urls:
            img = download_image_from_url(url)
            if img is None:
                continue

            parent_label = predict_parent_label_from_pil(img)
            if parent_label is None:
                continue

            parent_labels_used.add(parent_label)

            # subclass_meta에 등록
            register_subclass_label(parent_label, child_id, child_display)

            # 이미지 파일 저장
            path = save_sub_image(img, parent_label, child_id)
            saved_count += 1
            print(f"[AUTO] saved: parent={parent_label}, child={child_id}, path={path}")

        if saved_count > 0:
            processed_items.append(
                {
                    "index": idx,
                    "food_name": food_name,
                    "images_saved": saved_count,
                    "parent_labels": list(parent_labels_used),
                }
            )

    next_start = end if end < total else None

    return JSONResponse(
        {
            "message": "자동 수집 완료",
            "total": total,
            "start_index": start,
            "end_index": end,
            "processed": len(processed_items),
            "items": processed_items,
            "next_start_index": next_start,
        }
    )

# ---------------- API: 피드백 통계 ----------------

@app.get("/feedback_stats")
async def feedback_stats():
    """
    custom_data/ 아래에 쌓인 피드백 이미지 개수를 확인.
    """
    stats = count_feedback_samples()
    can_finetune = stats["total"] >= AUTO_FINETUNE_MIN_SAMPLES
    return JSONResponse({
        "total": stats["total"],
        "labels": stats["labels"],
        "can_finetune": can_finetune,
        "finetune_threshold": AUTO_FINETUNE_MIN_SAMPLES,
    })

# ---------------- API: finetune용 데이터 관리 (탭용) ----------------

@app.get("/finetune_data")
async def list_finetune_data():
    """
    finetune delete 탭에서 사용할 전체 데이터 리스트.
    - items: [{id, label, filename, path}, ...]
    - 이 id를 그대로 DELETE /finetune_data/item/{id} 에 쓰면 됨.
    """
    items = list_finetune_items()
    return JSONResponse({"total": len(items), "items": items})


@app.delete("/finetune_data/item/{item_id}")
async def delete_finetune_item_by_id(item_id: int):
    """
    GET /finetune_data 로 받은 items 중에서 특정 id를 가진 이미지를 삭제.
    - 프론트에서는 리스트에서 클릭한 항목의 id를 그대로 넘기면 됨.
    """
    items = list_finetune_items()
    if item_id < 0 or item_id >= len(items):
        return JSONResponse({"error": "해당 id의 항목이 없습니다."}, status_code=404)

    item = items[item_id]
    path = item["path"]
    label = item["label"]
    filename = item["filename"]

    if not os.path.exists(path):
        return JSONResponse({"error": "파일이 이미 삭제되었거나 존재하지 않습니다."}, status_code=404)

    try:
        os.remove(path)
        # 폴더가 비었으면 삭제
        label_dir = os.path.dirname(path)
        if os.path.isdir(label_dir) and len(os.listdir(label_dir)) == 0:
            os.rmdir(label_dir)
    except Exception as e:
        return JSONResponse({"error": f"삭제 중 오류: {e}"}, status_code=500)

    return JSONResponse({"message": "파일 삭제 완료", "id": item_id, "label": label, "filename": filename})


@app.delete("/finetune_data/label")
async def delete_finetune_label(label: str):
    """
    특정 라벨의 모든 finetune 이미지를 삭제.
    (이건 '이 라벨 전체 삭제' 버튼용)
    """
    label_dir = os.path.join(CUSTOM_DATA_DIR, label)
    if not os.path.exists(label_dir):
        return JSONResponse({"error": "해당 라벨 디렉토리가 존재하지 않습니다."}, status_code=404)

    removed_files = []
    try:
        for f in os.listdir(label_dir):
            path = os.path.join(label_dir, f)
            if os.path.isfile(path):
                os.remove(path)
                removed_files.append(f)
        try:
            os.rmdir(label_dir)
        except OSError:
            pass
    except Exception as e:
        return JSONResponse({"error": f"삭제 중 오류: {e}"}, status_code=500)

    return JSONResponse({"message": "라벨 데이터 삭제 완료", "label": label, "removed": removed_files})

# ---------------- API: 헤드 파인튜닝 (101 부모용) ----------------

@app.post("/finetune_head")
async def finetune_head():
    """
    custom_data/ 에 쌓인 사용자 라벨 데이터를 가지고
    EfficientNet 헤드(classifier.1)만 미세조정.
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
        print(f"[FINETUNE HEAD] Epoch {epoch+1}/{FINETUNE_EPOCHS} "
              f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return JSONResponse(
        {"message": "글로벌 헤드 파인튜닝 완료", "epochs": FINETUNE_EPOCHS}
    )

# ---------------- API: 부모별 서브헤드 파인튜닝 ----------------

@app.post("/finetune_subheads")
async def finetune_subheads():
    """
    sub_data/<parent>/<child>/*.jpg 를 사용해서
    부모별 feature-head(Linear(1280, n_child))를 학습한다.
    """
    meta = load_subclass_meta()
    if not meta:
        return JSONResponse({"message": "서브클래스 메타 데이터가 없습니다. feedback으로 먼저 서브라벨을 추가하세요."}, status_code=400)

    os.makedirs(SUBHEAD_DIR, exist_ok=True)
    train_results = []

    for parent_label, info in meta.items():
        children = info.get("children", [])
        child_ids = [c["id"] for c in children]
        if len(child_ids) < 2:
            print(f"[INFO] parent={parent_label} 는 서브클래스가 2개 미만이라 스킵")
            continue

        dataset = SubclassDataset(parent_label, child_ids, transform)
        if len(dataset) < len(child_ids) * 2:  # 너무 데이터가 적으면 스킵
            print(f"[INFO] parent={parent_label} 의 데이터({len(dataset)})가 너무 적어 스킵")
            continue

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        head = nn.Linear(1280, len(child_ids)).to(DEVICE)
        optimizer = torch.optim.Adam(head.parameters(), lr=SUBHEAD_LR)
        criterion = nn.CrossEntropyLoss()

        head.train()
        for ep in range(SUBHEAD_EPOCHS):
            running_loss = 0.0
            correct = total = 0
            for imgs, ys in dataloader:
                imgs = imgs.to(DEVICE)
                ys = ys.to(DEVICE)

                feats = extract_features(imgs)  # [B,1280], grad X (feats는 상수)
                optimizer.zero_grad()
                logits = head(feats)
                loss = criterion(logits, ys)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == ys).sum().item()
                total += ys.size(0)

            avg_loss = running_loss / len(dataloader)
            acc = correct / total if total > 0 else 0.0
            print(f"[SUBHEAD] parent={parent_label} Epoch {ep+1}/{SUBHEAD_EPOCHS} "
                  f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

        head_path = os.path.join(SUBHEAD_DIR, f"{parent_label}_head.pth")
        torch.save(head.state_dict(), head_path)
        if parent_label in _subhead_cache:
            del _subhead_cache[parent_label]

        train_results.append({
            "parent_label": parent_label,
            "children": child_ids,
            "samples": len(dataset),
        })

    if not train_results:
        return JSONResponse({"message": "학습할 서브헤드가 없거나, 데이터가 너무 적습니다."}, status_code=400)

    return JSONResponse({"message": "서브헤드 파인튜닝 완료", "results": train_results})

# ---------------- API: food_db 조회 ----------------

@app.get("/foods")
async def get_foods(limit: int = 100, offset: int = 0):
    """
    food_db에서 페이지네이션된 결과를 반환.
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
    - name에 한글이 있으면 영어로 번역해서 name으로 저장.
    """
    try:
        new_row, meta = add_food_row(
            name=food.name,
            carbs=food.carbs,
            protein=food.protein,
            fat=food.fat,
            serving_size=food.serving_size,
            unit=food.unit,
            calories=food.calories,
        )
        # ✅ 새 음식이 추가됐으니 텍스트 인덱스도 최신으로 갱신
        load_text_matcher()              # vectorizer 로드가 안 되어 있다면 먼저 로드
        rebuild_text_index_from_food_db()
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"DB 추가 중 오류: {e}"}, status_code=500)

    return JSONResponse(
        {
            "message": "새 음식이 food_db에 추가되었습니다.",
            "food": new_row,
            "meta": meta,
        }
    )

# ---------------- main: python server.py 로 바로 실행 가능 ----------------

if __name__ == "__main__":
    import uvicorn
    print("[INFO] 서버 시작: http://0.0.0.0:8000  (docs: /docs)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
