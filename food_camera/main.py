import os
import csv
from datetime import datetime
from typing import List
from difflib import get_close_matches

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from google.cloud import vision

FOOD_DB_PATH = "food_db.csv"
LOG_CSV_PATH = "intake_log.csv"

# -----------------------------
# 0. food_db.csv 로딩
# -----------------------------
def load_food_db(path: str):
    db = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 숫자 컬럼 float으로 변환
            row["serving_size"] = float(row["serving_size"])
            row["calories"] = float(row["calories"])
            row["protein"] = float(row["protein"])
            row["fat"] = float(row["fat"])
            row["carbs"] = float(row["carbs"])
            db.append(row)
    return db

FOOD_DB = load_food_db(FOOD_DB_PATH)


def search_food_db(query: str):
    """문자열 유사도로 FOOD_DB에서 가장 비슷한 음식 찾기."""
    if not FOOD_DB:
        return None

    names = [row["name"] for row in FOOD_DB]
    matches = get_close_matches(query, names, n=1, cutoff=0.4)
    if not matches:
        return None

    best_name = matches[0]
    for row in FOOD_DB:
        if row["name"] == best_name:
            return row
    return None


# -----------------------------
# 1. FastAPI 기본 세팅
# -----------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# 2. Vision API로 라벨 추출
# -----------------------------
def detect_food_labels(image_bytes: bytes, max_results: int = 5) -> List[str]:
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image, max_results=max_results)

    if response.error.message:
        raise Exception(response.error.message)

    labels = []
    for label in response.label_annotations:
        if label.score < 0.6:
            continue
        labels.append(label.description)
    return labels


@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    try:
        labels = detect_food_labels(img_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Vision API error: {str(e)}"},
        )

    top_labels = labels[:5]
    return {"labels": top_labels}


# -----------------------------
# 3. 로그 CSV에 기록하는 함수
# -----------------------------
def append_log(
    original_label: str,
    db_name: str,
    serving_qty: float,
    serving_unit: str,
    calories: float,
    protein: float,
    fat: float,
    carbs: float,
):
    """intake_log.csv에 한 줄 추가."""
    file_exists = os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # 헤더
            writer.writerow(
                [
                    "timestamp",
                    "original_label",
                    "db_name",
                    "serving_qty",
                    "serving_unit",
                    "calories",
                    "protein",
                    "fat",
                    "carbs",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                original_label,
                db_name,
                serving_qty,
                serving_unit,
                round(calories, 2),
                round(protein, 2),
                round(fat, 2),
                round(carbs, 2),
            ]
        )


# -----------------------------
# 4. 선택 음식 + 양 → 영양 계산 + 기록
# -----------------------------
@app.post("/get_calorie")
async def get_calorie(
    food_name: str = Form(...),   # Vision에서 고른 라벨
    amount: float = Form(...),    # 배수 (0.5~3.0 등)
):
    base = search_food_db(food_name)

    if base is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"'{food_name}' 에 해당하는 음식 정보를 DB에서 찾지 못했습니다."},
        )

    # food_db.csv 기준 정보
    base_serv_size = base["serving_size"]   # 예: 2 (tablespoon)
    base_unit = base["unit"]               # 예: tablespoon
    base_cal = base["calories"]
    base_prot = base["protein"]
    base_fat = base["fat"]
    base_carbs = base["carbs"]

    # amount 배만큼 최종 값 계산
    factor = amount
    total_serv = base_serv_size * factor
    total_cal = base_cal * factor
    total_prot = base_prot * factor
    total_fat = base_fat * factor
    total_carbs = base_carbs * factor

    # 로그 기록
    append_log(
        original_label=food_name,
        db_name=base["name"],
        serving_qty=total_serv,
        serving_unit=base_unit,
        calories=total_cal,
        protein=total_prot,
        fat=total_fat,
        carbs=total_carbs,
    )

    # 프론트에 반환
    return {
        "food_name": base["name"],
        "serving_qty": total_serv,
        "serving_unit": base_unit,
        "calories": total_cal,
        "protein": total_prot,
        "fat": total_fat,
        "carbs": total_carbs,
    }

# 실행:
# uvicorn main:app --reload
