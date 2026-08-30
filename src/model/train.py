import glob
import os
import random
from datetime import datetime, timezone

import boto3
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

load_dotenv()

from model import LSTMAutoencoder

VAL_RATIO = float(os.getenv("VAL_RATIO", "0.15"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "50"))
SEED = int(os.getenv("SEED", "42"))
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


def upload_checkpoint_to_s3(local_path: str, bucket: str, run_ts: datetime) -> str:
    # 체크포인트는 S3로 직접 업로드하고 W&B에는 S3 key(경로)만 텍스트로 전송
    key = (
        f"models/year={run_ts.strftime('%Y')}/month={run_ts.strftime('%m')}/day={run_ts.strftime('%d')}/"
        f"lstm_autoencoder_{run_ts.strftime('%H%M%S')}.ckpt"
    )
    s3_client = boto3.client("s3")
    s3_client.upload_file(local_path, bucket, key)
    print(f"🐇 Checkpoint uploaded to S3: s3://{bucket}/{key}")
    return key


def upload_scaler_to_s3(local_path: str, bucket: str, run_ts: datetime) -> str:
    key = (
        f"models/year={run_ts.strftime('%Y')}/month={run_ts.strftime('%m')}/day={run_ts.strftime('%d')}/"
        f"lstm_autoencoder_{run_ts.strftime('%H%M%S')}.json"
    )
    s3_client = boto3.client("s3")
    s3_client.upload_file(local_path, bucket, key)
    print(f"🐇 Scaler uploaded to S3: s3://{bucket}/{key}")
    return key


def main():
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.


if __name__ == "__main__":
    main()
