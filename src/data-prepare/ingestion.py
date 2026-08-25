import json
import os
from datetime import datetime, timezone
import boto3
import requests
from dotenv import load_dotenv

MIN_EXPECTED_RECORDS = 10_000

load_dotenv()
SPACETRACK_IDENTITY = os.getenv("SPACETRACK_IDENTITY")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def main():
    # 1. Space-Track login
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

    # 2. Fetch full TLE catalog (decayed objects excluded)
    query_url = (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        "DECAY_DATE/null-val/orderby/NORAD_CAT_ID/format/json"
    )
    response = session.get(query_url, timeout=120)

    if response.status_code != 200 or response.text.startswith("<!doctype"):
        raise RuntimeError(
            f"❌ API request failed (Status {response.status_code}): {response.text[:200]}"
        )

    tle_raw_json = response.text

    # 3. Data validation
    try:
        records = json.loads(tle_raw_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"❌ Response is not valid JSON: {e}") from e

    if not isinstance(records, list):
        raise RuntimeError(f"❌ Unexpected JSON shape (expected list, got {type(records).__name__})")

    record_count = len(records)
    if record_count < MIN_EXPECTED_RECORDS:
        raise RuntimeError(
            f"❌ Record count too low ({record_count} < {MIN_EXPECTED_RECORDS}); "
            "likely a truncated/failed catalog pull. Aborting upload."
        )

    required_fields = {"NORAD_CAT_ID", "EPOCH", "TLE_LINE1", "TLE_LINE2"}
    missing = required_fields - set(records[0].keys())
    if missing:
        raise RuntimeError(f"❌ Missing required fields in record: {missing}")

    # 4. S3 upload
    now = datetime.now(timezone.utc)
    raw_key = f"raw/year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/tle_raw_{now.strftime('%H%M%S')}.json"

    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=raw_key,
        Body=tle_raw_json,
        ContentType="application/json",
    )

    print(f"🐇 Raw TLE successfully ingested ({record_count} rows): s3://{S3_BUCKET_NAME}/{raw_key}")
    print(raw_key)  # XCom value

if __name__ == "__main__":
    main()
