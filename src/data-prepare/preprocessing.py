import io
import json
import os
import re
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

load_dotenv()
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
RAW_S3_KEY = os.getenv("RAW_S3_KEY")

# 1차 screening engine 임계값: 실제 정밀 충돌위험 계산(3~4단계)의 앞단으로 후보군을 넉넉하게 골라내는 용도
CONJUNCTION_SCREENING_THRESHOLD_KM = float(
    os.getenv("CONJUNCTION_SCREENING_THRESHOLD_KM", "25.0")
)

# 심우주/달 궤도 미션(ARTEMIS, Chang'e, Chandrayaan, DRO 등) 제외
# GEO 고도(~35,786km) + HEO(Molniya/Tundra 등, apogee 최대 ~46,000km)까지는 유지
DEEP_SPACE_ALT_THRESHOLD_KM = float(
    os.getenv("DEEP_SPACE_ALT_THRESHOLD_KM", "50000.0")
)

# 변동치 비교 기준 시간 간격
DEVIATION_LOOKBACK_HOURS = float(os.getenv("DEVIATION_LOOKBACK_HOURS", "24.0"))

# dt_hours가 최소 간격 미만이면 변화율 계산 자체를 하지 않고 NaN 처리
MIN_DT_HOURS_FOR_DEVIATION = float(os.getenv("MIN_DT_HOURS_FOR_DEVIATION", "1.0"))

# Space-Track GP(JSON) 응답에서 그대로 가져올 필드
RAW_COLUMNS = [
    "NORAD_CAT_ID",
    "OBJECT_NAME",
    "EPOCH",
    "INCLINATION",
    "ECCENTRICITY",
    "ARG_OF_PERICENTER",
    "RA_OF_ASC_NODE",
    "MEAN_MOTION",
    "MEAN_ANOMALY",
    "BSTAR",
    "TLE_LINE1",
    "TLE_LINE2",
]
NUMERIC_COLUMNS = [
    "INCLINATION", "ECCENTRICITY", "ARG_OF_PERICENTER",
    "RA_OF_ASC_NODE", "MEAN_MOTION", "MEAN_ANOMALY", "BSTAR",
]

# 변동치 계산 대상 궤도 요소. 각도(0~360도로 wrap되는) 컬럼은 ANGLE_COLUMNS에 별도 표시
DEVIATION_ELEMENTS = [
    "INCLINATION", "ECCENTRICITY", "ARG_OF_PERICENTER",
    "RA_OF_ASC_NODE", "MEAN_MOTION", "BSTAR",
]
ANGLE_COLUMNS = {"ARG_OF_PERICENTER", "RA_OF_ASC_NODE"}

# 서로 단위가 다른 궤도 요소를 하나의 스코어로 합치기 위한 정규화 스케일 (휴리스틱 초기값)
# ⚠️ ORBITAL_DEVIATION_METRIC(아래 DEVIATION_SCALES 기반 가중합)은 3단계 모델 학습 입력으로는 사용 안함:
# 신규 발사 위성처럼 BSTAR 추정이 아직 불안정한 객체에서 BSTAR 항이 전체 점수를 100% 지배하는 문제 확인됨 (2026-08-27, QIANFAN 계열에서 발견)
# 모델 입력은 DELTA_*_PER_HR 원소별 컬럼을 그대로(혹은 학습셋 기준 정규화해서) 사용할 것. 이 합산 점수는 대략적인 모니터링/스크리닝 참고용으로만 유지
DEVIATION_SCALES = {
    "INCLINATION": 1.0,        # deg
    "ECCENTRICITY": 0.01,
    "ARG_OF_PERICENTER": 1.0,  # deg
    "RA_OF_ASC_NODE": 1.0,     # deg
    "MEAN_MOTION": 0.01,       # rev/day
    "BSTAR": 1e-4,
}

def extract_date_partition(raw_key: str) -> tuple[str, str, str]:
    # raw/year=2026/month=08/day=22/tle_raw_020100.json → ('2026','08','22')
    match = re.search(r"year=(\d{4})/month=(\d{2})/day=(\d{2})", raw_key)
    if not match:
        raise ValueError(f"❌ Could not find date partition in raw key: {raw_key}")
    return match.group(1), match.group(2), match.group(3)


def compute_eci_state(line1: str, line2: str, name: str, ts) -> dict:
    # TLE epoch 시점의 ECI(GCRS) 위치/속도 벡터 계산 (sgp4 propagation via skyfield)
    sat = EarthSatellite(line1, line2, name, ts)
    geocentric = sat.at(sat.epoch)
    x, y, z = geocentric.position.km
    vx, vy, vz = geocentric.velocity.km_per_s

    # ECEF (지구 고정 좌표계) + LLA (위도/경도/고도)
    ecef_x, ecef_y, ecef_z = geocentric.frame_xyz(itrs).km
    subpoint = wgs84.subpoint(geocentric)

    return {
        "X_KM": x, "Y_KM": y, "Z_KM": z,
        "VX_KM_S": vx, "VY_KM_S": vy, "VZ_KM_S": vz,
        "ECEF_X_KM": ecef_x, "ECEF_Y_KM": ecef_y, "ECEF_Z_KM": ecef_z,
        "LAT_DEG": subpoint.latitude.degrees,
        "LON_DEG": subpoint.longitude.degrees,
        "ALT_KM": subpoint.elevation.km,
    }


def build_dataframe(records: list[dict], ts) -> pd.DataFrame:
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.


def _angular_delta(curr: pd.Series, prev: pd.Series) -> pd.Series:
    # 0~360도 wrap-around를 고려한 최단 각도 차 (-180, 180] 범위로 반환
    diff = (curr - prev + 180) % 360 - 180
    return diff


def compute_orbital_deviation(df: pd.DataFrame, prev_df: pd.DataFrame | None) -> pd.DataFrame:
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.


def compute_conjunction_screening(df: pd.DataFrame, threshold_km: float = CONJUNCTION_SCREENING_THRESHOLD_KM) -> pd.DataFrame:
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.


def get_reference_processed_df(s3_client, bucket: str, target_time: datetime, tolerance_days: int = 1) -> pd.DataFrame | None:
    # target_time에 가장 가까운 processed 파일을 찾아 로드
    candidate_dates = {
        (target_time + timedelta(days=d)).date() for d in range(-tolerance_days, tolerance_days + 1)
    }
    prefixes = [
        f"processed/year={d.year:04d}/month={d.month:02d}/day={d.day:02d}/" for d in candidate_dates
    ]

    candidates = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            candidates.extend(page.get("Contents", []))

    if not candidates:
        return None

    best = min(candidates, key=lambda o: abs((o["LastModified"] - target_time).total_seconds()))
    gap_hours = abs((best["LastModified"] - target_time).total_seconds()) / 3600.0

    print(
        f"🐇 Loading reference processed snapshot: s3://{bucket}/{best['Key']} "
        f"(target {target_time.isoformat()}, actual gap {gap_hours:.1f}h)"
    )
    obj = s3_client.get_object(Bucket=bucket, Key=best["Key"])
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def main():
    if not RAW_S3_KEY:
        raise RuntimeError("❌ RAW_S3_KEY environment variable is missing. (Check Airflow XCom transmission)")
    if not S3_BUCKET_NAME:
        raise RuntimeError("❌ S3_BUCKET_NAME environment variable is missing.")

    s3_client = boto3.client("s3")

    # 1. raw JSON download
    obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=RAW_S3_KEY)
    records = json.loads(obj["Body"].read())
    if not records:
        print(f"⚠️ No records found in raw file: {RAW_S3_KEY}")
        return

    # 2. ECI/ECEF/LLA 계산 (leap second/deltat 파일 다운로드 없이 오프라인 동작)
    ts = load.timescale(builtin=True)
    df = build_dataframe(records, ts)
    if df.empty:
        print("⚠️ No successfully converted records. Skipping Parquet generation.")
        return

    # 3. 데이터 검증
    df = validate_dataframe(df)

    # 4. 궤도 요소 변동치 (약 DEVIATION_LOOKBACK_HOURS 전 스냅샷과 비교)
    year, month, day = extract_date_partition(RAW_S3_KEY)
    target_time = datetime.now(timezone.utc) - timedelta(hours=DEVIATION_LOOKBACK_HOURS)
    prev_df = get_reference_processed_df(s3_client, S3_BUCKET_NAME, target_time)
    df = compute_orbital_deviation(df, prev_df)

    # 5. 상대 거리 1차 screening
    df = compute_conjunction_screening(df)

    # 6. Parquet 변환
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)

    # 7. processed 적재
    now = datetime.now(timezone.utc)
    processed_key = (
        f"processed/year={year}/month={month}/day={day}/"
        f"tle_processed_{now.strftime('%H%M%S')}.parquet"
    )

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=processed_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"🐇 Processed TLE successfully saved ({len(df)} rows): s3://{S3_BUCKET_NAME}/{processed_key}")


if __name__ == "__main__":
    main()
