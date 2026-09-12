from __future__ import annotations

import json
import os
import subprocess
import sys

import boto3

NAMESPACE = os.getenv("K8S_NAMESPACE", "ftor")
DEPLOYMENT = os.getenv("K8S_DEPLOYMENT", "model-serving")
ANNOTATION_KEY = "ftor.io/checkpoint-key"
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "models/")
ROLLOUT_TIMEOUT = os.getenv("ROLLOUT_TIMEOUT", "180s")


def find_latest_checkpoint_key(bucket: str, prefix: str) -> str:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    candidates = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        candidates.extend(o for o in page.get("Contents", []) if o["Key"].endswith(".ckpt"))
    if not candidates:
        raise RuntimeError(f"❌ No checkpoint found in S3: s3://{bucket}/{prefix}")
    best = max(candidates, key=lambda o: o["LastModified"])
    return best["Key"]


def get_current_annotation() -> str | None:
    result = subprocess.run(
        [
            "kubectl", "get", "deployment", DEPLOYMENT,
            "-n", NAMESPACE,
            "-o", f"jsonpath={{.spec.template.metadata.annotations.{ANNOTATION_KEY}}}",
        ],
        capture_output=True, text=True, check=True,
    )
    value = result.stdout.strip()
    return value or None


def patch_annotation(checkpoint_key: str) -> None:
    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {ANNOTATION_KEY: checkpoint_key}
                }
            }
        }
    }
    subprocess.run(
        ["kubectl", "patch", "deployment", DEPLOYMENT, "-n", NAMESPACE, "-p", json.dumps(patch)],
        check=True,
    )


def wait_for_rollout() -> bool:
    result = subprocess.run(
        [
            "kubectl", "rollout", "status", f"deployment/{DEPLOYMENT}",
            "-n", NAMESPACE, f"--timeout={ROLLOUT_TIMEOUT}",
        ],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def rollback() -> None:
    subprocess.run(["kubectl", "rollout", "undo", f"deployment/{DEPLOYMENT}", "-n", NAMESPACE], check=True)


def main() -> None:
    if not S3_BUCKET_NAME:
        raise RuntimeError("❌ S3_BUCKET_NAME environment variable is required.")

    latest_key = find_latest_checkpoint_key(S3_BUCKET_NAME, MODEL_S3_PREFIX)
    current = get_current_annotation()

    if current == latest_key:
        print(f"🐇 Already up to date: {latest_key}")
        return

    print(f"🐇 New checkpoint detected: {current!r} -> {latest_key!r}")
    patch_annotation(latest_key)

    if wait_for_rollout():
        print(f"🐇 Rollout succeeded: {latest_key}")
    else:
        print("❌ Rollout failed within timeout, rolling back to previous revision.", file=sys.stderr)
        rollback()
        sys.exit(1)


if __name__ == "__main__":
    main()
