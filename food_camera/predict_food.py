# predict_food.py
import os
import argparse

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import Food101
from PIL import Image
import pandas as pd
from difflib import get_close_matches

# --------- 경로 설정 ---------
MODEL_PATH = "food_classifier.pth"
CLASS_NAMES_PATH = "class_names.txt"  # 없거나 이상하면 다시 생성해줌
FOOD_DB_PATH = "food_db.csv"          # 우리가 만든 영양 DB
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- Food-101에서 클래스 이름 가져오기 ---------
def get_food101_class_names():
    """
    훈련 때와 동일하게 Food101(root='./data', split='train') 기준으로
    클래스 이름 목록을 가져온다. (이미 다운로드 되어 있으면 바로 사용)
    """
    ds = Food101(root="./data", split="train", download=False)
    return list(ds.classes)


def ensure_class_names():
    """
    class_names.txt가 깨져 있거나(1줄) 없으면,
    Food-101 메타데이터에서 다시 생성한다.
    """
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        if len(names) == 101:  # Food-101은 101 클래스
            print(f"📂 기존 class_names.txt 사용 (클래스 수: {len(names)})")
            return names
        else:
            print(f"⚠ class_names.txt 클래스 수가 이상함({len(names)}). Food-101 기준으로 재생성할게요.")

    # 여기까지 오면: 파일이 없거나 이상함 → Food101에서 다시 얻기
    names = get_food101_class_names()
    print(f"✅ Food-101에서 클래스 이름 {len(names)}개 로드")

    # 안전하게 class_names.txt도 다시 써줌
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")
    print("✅ class_names.txt 재생성 완료")

    return names


# --------- 모델 로드 ---------
def load_model(num_classes_expected: int):
    """
    EfficientNet-B0 구조를 만들고, 체크포인트에서 가중치를 로드한다.
    체크포인트의 마지막 레이어 출력 차원과 Food-101 클래스 수가 맞는지 확인.
    """
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # 체크포인트에 저장된 classifier.1.weight의 shape로 클래스 수 확인
    num_classes_ckpt = state_dict["classifier.1.weight"].shape[0]
    print(f"🔎 체크포인트 기준 클래스 수: {num_classes_ckpt}")

    if num_classes_ckpt != num_classes_expected:
        print(f"⚠ 경고: Food-101 클래스 수({num_classes_expected})와 "
              f"체크포인트 클래스 수({num_classes_ckpt})가 다릅니다.")
        print("   → 훈련 코드와 예측 코드의 class_names가 달라졌을 수 있어요.")

    # EfficientNet-B0 구조 생성
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes_ckpt)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, num_classes_ckpt


# --------- 이미지 전처리 (훈련 때와 동일) ---------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def preprocess_image(img_path: str):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor.to(DEVICE)


# --------- 예측 ---------
def predict_image(model, class_names, img_path: str, topk: int = 3):
    tensor = preprocess_image(img_path)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # [num_classes]
        top_probs, top_idxs = torch.topk(probs, k=topk)

    results = []
    for p, idx in zip(top_probs, top_idxs):
        idx = idx.item()
        conf = float(p.item())
        if 0 <= idx < len(class_names):
            label = class_names[idx]
        else:
            label = f"unknown_{idx}"
        results.append((label, conf))
    return results


# --------- 영양 DB 로드 & 조회 ---------
def load_food_db():
    if not os.path.exists(FOOD_DB_PATH):
        print(f"⚠ {FOOD_DB_PATH} 파일을 찾을 수 없습니다. 영양정보는 조회하지 못해요.")
        return None
    df = pd.read_csv(FOOD_DB_PATH)
    return df


def lookup_nutrition(food_db: pd.DataFrame, name_en: str):
    """
    1) 정확 매칭
    2) 소문자 매칭
    3) 문자열 유사도(top-1)
    순서로 food_db에서 찾아본다.
    """
    if food_db is None:
        return None

    # 1) 정확 매칭
    row = food_db[food_db["name"] == name_en]
    if len(row) == 0:
        # 2) lower-case 매칭
        lower_name = name_en.lower()
        names_lower = food_db["name"].astype(str).str.lower()
        row = food_db[names_lower == lower_name]

    if len(row) == 0:
        # 3) 유사도 기반 근접 매칭
        candidates = list(food_db["name"].astype(str).unique())
        match = get_close_matches(name_en, candidates, n=1, cutoff=0.6)
        if not match:
            return None
        row = food_db[food_db["name"] == match[0]]

    row = row.iloc[0]
    info = {
        "name": row["name"],
        "serving_size": float(row.get("serving_size", 1.0)),
        "unit": row.get("unit", ""),
        "calories": float(row.get("calories", 0.0)),
        "protein": float(row.get("protein", 0.0)),
        "fat": float(row.get("fat", 0.0)),
        "carbs": float(row.get("carbs", 0.0)),
    }
    return info


# --------- 메인 ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="예측할 음식 이미지 경로")
    parser.add_argument("--topk", type=int, default=3, help="상위 몇 개 후보를 볼지")
    args = parser.parse_args()

    # 1) 클래스 이름 확보 (Food-101 기준)
    class_names = ensure_class_names()
    num_classes_expected = len(class_names)
    print(f"📂 최종 클래스 수: {num_classes_expected}")

    # 2) 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ {MODEL_PATH} 를 찾을 수 없습니다. 학습 스크립트 먼저 돌려서 모델을 만들어야 해요.")
        return

    print("✅ 모델 로딩 중...")
    model, num_classes_ckpt = load_model(num_classes_expected)

    # 3) food_db 로드
    print("✅ food_db 로딩 중...")
    food_db = load_food_db()

    # 4) 이미지 예측
    print(f"📷 예측할 이미지: {args.image}")
    preds = predict_image(model, class_names, args.image, topk=args.topk)

    print("\n🔎 Top 예측 결과:")
    for i, (label, conf) in enumerate(preds, start=1):
        print(f"{i}. {label} (신뢰도: {conf:.3f})")

    # 5) 가장 높은 후보로 영양정보 조회
    best_label, best_conf = preds[0]
    print(f"\n🍽 최종 선택: {best_label} (신뢰도 {best_conf:.3f})")

    nutri = lookup_nutrition(food_db, best_label)
    if nutri is None:
        print("⚠ 이 음식 이름으로는 food_db.csv에서 영양 정보를 찾지 못했습니다.")
        print("   → Food-101 클래스 이름과 food_db의 name 컬럼 사이 매핑을 조금씩 맞춰줘야 해요.")
    else:
        print("\n📊 기본 1회 제공량 기준 영양 정보:")
        print(f" - 기준량: {nutri['serving_size']} {nutri['unit']}")
        print(f" - 칼로리: {nutri['calories']} kcal")
        print(f" - 탄수화물: {nutri['carbs']} g")
        print(f" - 단백질: {nutri['protein']} g")
        print(f" - 지방: {nutri['fat']} g")


if __name__ == "__main__":
    main()
