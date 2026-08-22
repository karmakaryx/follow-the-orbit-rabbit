import os
from datetime import datetime, timezone
import boto3
import requests
from dotenv import load_dotenv

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

    # 2. 최신 TLE 100건 조회
    query_url = (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        "EPOCH/>now-3/orderby/NORAD_CAT_ID/limit/100/format/json"
    )
    response = session.get(query_url, timeout=30)

    # 응답 검증 (JSON이 맞는지 확인)
    if response.status_code != 200 or response.text.startswith("<!doctype"):
        raise RuntimeError(
            f"❌ API 요청 실패 (Status: {response.status_code}): {response.text[:200]}"
        )

    tle_raw_json = response.text

    # 3. S3 upload
    now = datetime.now(timezone.utc)
    s3_key = f"raw/year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/tle_raw_{now.strftime('%H%M%S')}.json"

    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Body=tle_raw_json,
        ContentType="application/json",
    )

    print(f"🐇 Raw TLE S3 적재 성공: s3://{S3_BUCKET_NAME}/{s3_key}")

if __name__ == "__main__":
    main()
