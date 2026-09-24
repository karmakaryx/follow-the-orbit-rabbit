import json
import os
import time
from datetime import datetime, timezone

import boto3
import requests
from dotenv import load_dotenv

MIN_EXPECTED_RECORDS = 10_000

load_dotenv()
SPACETRACK_IDENTITY = os.getenv("SPACETRACK_IDENTITY")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
RAW_LOCAL_RETENTION_HOURS = float(os.getenv("RAW_LOCAL_RETENTION_HOURS", "6"))


def save_raw_locally(content: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, local_path)


def cleanup_old_local_files(root: str, max_age_hours: float) -> None:
    if not os.path.isdir(root):
        return

    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError as e:
                print(f"⚠️ Failed to remove old local file {path}: {e}")

        if dirpath != root:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass

    if removed:
        print(f"🐇 Cleaned up {removed} local file(s) older than {max_age_hours}h under {root}")


def main():
    # 1. 마운트 누락 방지 경고
    if not os.path.ismount(DATA_DIR):
        print(f"⚠️ {DATA_DIR} is not a mounted volume. Local raw copy will not be visible to other containers.")

    # 2. Space-Track login
    session = requests.Session()
    login_url = "https://www.space-track.org/ajaxauth/login"
    session.post(
        login_url,
        data={
            "identity": SPACETRACK_IDENTITY,
            "password": SPACETRACK_PASSWORD,
        },
        timeout=30,
    )

    # 3. Fetch full TLE catalog (decayed objects excluded)
    query_url = (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        "DECAY_DATE/null-val/orderby/NORAD_CAT_ID/format/json"
    )
    response = session.get(query_url, timeout=120)

    if response.status_code != 200 or response.text.startswith("<!doctype"):
        raise RuntimeError(f"❌ API request failed (Status {response.status_code}): {response.text[:200]}")

    tle_raw_json = response.text

    # 4. Data validation
    try:
        records = json.loads(tle_raw_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"❌ Response is not valid JSON: {e}") from e

    if not isinstance(records, list):
        raise RuntimeError(f"❌ Unexpected JSON shape (expected list, got {type(records).__name__})")

    record_count = len(records)
    if record_count < MIN_EXPECTED_RECORDS:
        raise RuntimeError(f"❌ Record count too low ({record_count} < {MIN_EXPECTED_RECORDS}). Aborting upload.")

    required_fields = {"NORAD_CAT_ID", "EPOCH", "TLE_LINE1", "TLE_LINE2"}
    missing = required_fields - set(records[0].keys())
    if missing:
        raise RuntimeError(f"❌ Missing required fields in record: {missing}")

    # 5. 로컬 저장 (정제가 실패해도 Space-Track을 다시 호출하지 않고 재시도 가능)
    now = datetime.now(timezone.utc)
    raw_key = f"raw/year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/tle_raw_{now.strftime('%H%M%S')}.json"
    local_raw_path = os.path.join(DATA_DIR, raw_key)
    save_raw_locally(tle_raw_json, local_raw_path)
    print(f"🐇 Raw TLE saved locally: {local_raw_path}")

    # 6. S3 원본 백업 업로드
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        local_raw_path,
        S3_BUCKET_NAME,
        raw_key,
        ExtraArgs={"ContentType": "application/json"},
    )

    # 7. 보관 기간이 지난 로컬 raw 파일 자동 삭제
    cleanup_old_local_files(os.path.join(DATA_DIR, "raw"), RAW_LOCAL_RETENTION_HOURS)

    print(f"🐇 Raw TLE successfully ingested ({record_count} rows): s3://{S3_BUCKET_NAME}/{raw_key}")
    print(raw_key)  # XCom value


if __name__ == "__main__":
    main()
