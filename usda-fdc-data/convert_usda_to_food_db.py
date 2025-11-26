import pandas as pd
import numpy as np

# ★ 여기를 본인 CSV 경로에 맞게 수정하세요
INPUT_CSV = "data/usda_food_nutrition_data.csv"   # usda-fdc-data가 만든 파일
OUTPUT_CSV = "food_db.csv"                        # 우리가 쓸 최종 파일


def pick_name(row):
    """
    food_common_name이 있으면 그걸 우선 사용,
    없거나 'no_value'면 food_description 사용
    """
    common = row.get("food_common_name", "")
    desc = row.get("food_description", "")

    if isinstance(common, str) and common.strip() and common.lower() != "no_value":
        return common.strip()
    return str(desc).strip()


def main():
    print(f"🔍 입력 CSV 읽는 중: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV)

    # 1) 이름 컬럼 만들기
    df_raw["name"] = df_raw.apply(pick_name, axis=1)

    # 숫자 컬럼들 안전하게 float로 변환
    num_cols = [
        "portion_gram_weight",
        "portion_energy",
        "carbohydrate_by_difference",
        "protein",
        "total_lipid_fat",
        "portion_amount",
    ]
    for col in num_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        else:
            df_raw[col] = np.nan

    g = df_raw["portion_gram_weight"]
    kcal = df_raw["portion_energy"]

    # 2) 기본 kcal는 portion_energy 그대로 사용
    df_raw["calories"] = kcal

    # 3) 1차로 g 단위 탄단지 추정 (밀도[g/g] * g)
    carb_density = df_raw["carbohydrate_by_difference"]
    prot_density = df_raw["protein"]
    fat_density  = df_raw["total_lipid_fat"]

    carb_g_raw = carb_density * g
    prot_g_raw = prot_density * g
    fat_g_raw  = fat_density  * g

    # 4) kcal 역산/보정
    #   - 우선 raw g 기준으로 kcal 계산
    carb_kcal_raw = carb_g_raw * 4.0
    prot_kcal_raw = prot_g_raw * 4.0
    fat_kcal_raw  = fat_g_raw  * 9.0
    macro_kcal_sum = carb_kcal_raw + prot_kcal_raw + fat_kcal_raw

    #   - 보정된 g 값을 담을 컬럼 초기화
    carb_g_adj = pd.Series(np.zeros(len(df_raw)), dtype=float)
    prot_g_adj = pd.Series(np.zeros(len(df_raw)), dtype=float)
    fat_g_adj  = pd.Series(np.zeros(len(df_raw)), dtype=float)

    # (1) macro_kcal_sum > 0 이고, calories > 0 인 행: 스케일링으로 보정
    mask_scale = (macro_kcal_sum > 0) & (kcal > 0)
    scale = pd.Series(np.zeros(len(df_raw)), dtype=float)
    scale[mask_scale] = (kcal[mask_scale] / macro_kcal_sum[mask_scale]).astype(float)

    carb_g_adj[mask_scale] = (carb_g_raw[mask_scale] * scale[mask_scale]).astype(float)
    prot_g_adj[mask_scale] = (prot_g_raw[mask_scale] * scale[mask_scale]).astype(float)
    fat_g_adj[mask_scale]  = (fat_g_raw[mask_scale]  * scale[mask_scale]).astype(float)

    # (2) macro_kcal_sum == 0 이거나 kcal <= 0 인 행: 기본 비율로 분배
    #     예시 비율: 탄수 50%, 단백질 20%, 지방 30%
    mask_fallback = ~mask_scale & (kcal > 0)
    carb_ratio = 0.5
    prot_ratio = 0.2
    fat_ratio  = 0.3

    carb_g_adj[mask_fallback] = (kcal[mask_fallback] * carb_ratio / 4.0).astype(float)
    prot_g_adj[mask_fallback] = (kcal[mask_fallback] * prot_ratio / 4.0).astype(float)
    fat_g_adj[mask_fallback]  = (kcal[mask_fallback] * fat_ratio  / 9.0).astype(float)

    # 5) 우리가 쓸 컬럼만 뽑아서 이름 맞추기
    df = pd.DataFrame()
    df["name"] = df_raw["name"].astype(str)
    df["serving_size"] = df_raw["portion_amount"]
    df["unit"] = df_raw["portion_unit"].astype(str)
    df["calories"] = df_raw["calories"].astype(float)
    df["protein"] = prot_g_adj.astype(float)
    df["fat"] = fat_g_adj.astype(float)
    df["carbs"] = carb_g_adj.astype(float)

    # 6) 기본적인 정리/필터링
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["name", "serving_size", "unit", "calories"])

    df = df[df["unit"].str.lower() != "no_value"]
    df = df[df["serving_size"] > 0]
    df = df[df["calories"] > 0]

    # 너무 작은 숫자/소수점 깔끔하게 정리
    df["calories"] = df["calories"].round(2)
    df["protein"] = df["protein"].round(2)
    df["fat"] = df["fat"].round(2)
    df["carbs"] = df["carbs"].round(2)

    # 7) 중복 제거 (같은 음식, 같은 서빙 단위)
    df = df.drop_duplicates(subset=["name", "serving_size", "unit"])

    print(f"✅ 최종 행 개수: {len(df)}")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"💾 저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
