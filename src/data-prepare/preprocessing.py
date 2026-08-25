import io
import json
import os
import re
from datetime import datetime, timezone
import boto3
import pandas as pd
from dotenv import load_dotenv
from skyfield.api import load, EarthSatellite

load_dotenv()
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
RAW_S3_KEY = os.getenv("RAW_S3_KEY")

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

def extract_date_partition(raw_key: str) -> tuple[str, str, str]:
    # raw/year=2026/month=08/day=22/tle_raw_020100.json -> ('2026','08','22')
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
    return {
        "X_KM": x, "Y_KM": y, "Z_KM": z,
        "VX_KM_S": vx, "VY_KM_S": vy, "VZ_KM_S": vz,
    }

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

    # 2. TLE epoch 시점 ECI 상태벡터 계산 (leap second/deltat 파일 다운로드 없이 오프라인 동작)
    ts = load.timescale(builtin=True)
    rows = []
    for rec in records:
        line1 = rec.get("TLE_LINE1")
        line2 = rec.get("TLE_LINE2")
        if not line1 or not line2:
            continue

        try:
            eci = compute_eci_state(line1, line2, rec.get("OBJECT_NAME", ""), ts)
        except Exception as e:
            print(f"⚠️ SGP4 propagation failed (NORAD {rec.get('NORAD_CAT_ID')}): {e}")
            continue

        row = {col: rec.get(col) for col in RAW_COLUMNS}
        row.update(eci)
        rows.append(row)

    if not rows:
        print("⚠️ No successfully converted records. Skipping Parquet generation.")
        return

    # 3. 스키마 정리
    df = pd.DataFrame(rows)
    df["EPOCH"] = pd.to_datetime(df["EPOCH"])
    df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"], errors="coerce").astype("Int64")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Parquet 변환
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)

    # 5. processed 적재
    year, month, day = extract_date_partition(RAW_S3_KEY)
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
