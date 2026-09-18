import json
import os
import urllib.request
import boto3

s3 = boto3.client("s3")


def handler(event, context):
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = urllib.request.unquote(event["Records"][0]["s3"]["object"]["key"])

    if not (key.endswith(".ckpt") or key.endswith(".json")):
        print(f"⚠️ Ignoring unrelated key: {key}")
        return

    stem = key.rsplit(".", 1)[0]  # .../lstm_autoencoder_020100
    ckpt_key = f"{stem}.ckpt"
    scaler_key = f"{stem}.json"

    if not _exists(bucket, ckpt_key) or not _exists(bucket, scaler_key):
        print(f"🐇 Pair not complete yet for {stem}, skipping dispatch.")
        return

    print(f"🐇 Checkpoint pair confirmed: {ckpt_key} + {scaler_key}")
    _trigger_deploy_workflow()


def _exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def _trigger_deploy_workflow():
    owner = os.environ["GITHUB_OWNER"]
    repo = os.environ["GITHUB_REPO"]
    ref = os.environ.get("GITHUB_REF", "phase2")
    token = os.environ["GITHUB_PAT"]

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/deploy-model.yml/dispatches"
    req = urllib.request.Request(
        url,
        data=json.dumps({"ref": ref}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 204:
            raise RuntimeError(f"❌ Failed to dispatch workflow (status {resp.status})")
    print(f"🐇 deploy-model.yml workflow_dispatch triggered on ref={ref}")
