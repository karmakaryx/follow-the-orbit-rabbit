![banner_ftor](./assets/banner_ftor.jpg)

<div align="center">
  <h3>Space-Track TLE Data Pipeline & Collision Avoidance MLOps System using Airflow, MLflow, Kubernetes and FastAPI</h3>
</div>

## **🎥 Feature Highlights**
🚨 NOTICE: Phases 1 thru 5 will be developed sequentially. Phase 1 is currently open as a prerelease. 🚨
<p align="center">
  <video src="https://github.com/user-attachments/assets/c97abd70-ae9c-49ca-a059-9b3cac9e0629" width="100%" controls autoplay muted loop>
    Your browser does not support the video tag.
  </video>
</p>

<p align="center">
  [ <a href="./README.md">English</a> | <a href="./README_KR.md">한국어</a> ]
</p>

## **🛰️ Project Info**
### Project Objectives
- Detect anomalies in collision risks among satellites, launch vehicles, and space debris.
- Build a primary collision alert MLOps system combining TLE-based orbital element variation detection with spatial proximity screening.
- Develop ML models to predict detailed Probability of Collision (PoC) for selected high-risk targets.

### Data Collection
- Acquire orbital elements (TLE) and location data for Earth-orbiting satellites and space debris from Space-Track.org (CSpOC).
- GP (TLE) data is updated only about 2 to 4 times a day for LEO; the recommended API polling frequency is once per hour.
- API Rate Limits: Max 30 requests per minute / Max 300 requests per hour
- Prediction Range Limits: Due to error accumulation inherent to SGP4/TLE, the system focuses on short-term (within 3 to 7 days) primary collision risk alerts

### Tech Stack
> 💡 **Note:** Additional specs and components are currently under design.
- **Airflow:** Data Pipeline & Automated Orchestration
- **S3:** Data Lake & Artifact Storage
- **Docker:** Task-Isolated Experimentation Infrastructure
- **PyTorch Lightning:** Model Training
- **W&B:** Experiment Tracking & Performance Monitoring
- **MLflow:** Model Registry
- **FastAPI:** Inference Serving
- **Minikube:** Local Kubernetes Cluster (EKS Alternative)
- **AWS Lambda:** Serverless Event Trigger
- **GitHub Actions:** CI/CD Pipeline
- **Amazon SQS:** Message Queue
- **Streamlit:** Dashboard

---

## **🎬 MLOps Scenario**
### STEP 1. [수집] Data Ingestion (Airflow)
- Poll Space-Track REST API.
- Periodically perform incremental ingestion of the full TLE catalog (~35,000 objects) for launch vehicles and satellite constellations of interest (e.g. Starlink, LEO space debris).
- Collect active objects only ((where decay date is empty)
- Store raw JSON in S3 (e.g. `s3://my-bucket/raw/year=2026/month=08/day=22/tle_raw_020100.json`)

### STEP 2. [전처리/가공] Preprocessing & Feature Engineering
- Coordinate transformation using SGP4 and Skyfield libraries: TLE → ECI → ECEF → LLA
- Data Validation: Schema verification, missing value checks, and value range validation
- Deep Space & Lunar Orbit Filtering: Exclude objects with ALT_KM > 50,000 km, while retaining GEO and HEO
- Orbital Element Variations: Calculate anomalous rate-of-change features based on inclination($i$), eccentricity($e$), argument of pericenter($\omega$), mean motion($n$), etc.
- Screening Engine: Primary filtering by orbital similarity groups using a KDTree-based spatial index to rapidly extract proximity candidates instead of evaluating all pairwise combinations
- Relative Distance Calculation: Identify potential collision risks by evaluating whether the relative distance between two objects falls within a defined proximity threshold
- Convert data to Parquet format and store in S3 processed path (e.g. `s3://my-bucket/processed/year=2026/month=08/day=22/tle_processed_020100.parquet`)

### STEP 3. [학습/등록] Model Training & Registry (LSTM Autoencoder/W&B)
**1. Time-Series Sequence Construction**
- Aggregate multiple processed parquet snapshots to construct time-series windows per object (`NORAD_CAT_ID`). Designed to start with short windows initially and scale up via environment variables as data accumulates over time.
- Merge multiple Parquet files into a single DataFrame and deduplicate (NORAD_CAT_ID, EPOCH) pairs.
- Exclude records with epochs older than `TLE_MAX_AGE_DAYS` (default: 30 days) to filter out long-outdated objects (e.g. stale for decades).
- Extract the recent `SEQUENCE_WINDOW_HOURS` segment (default: 3 days) per object. If snapshots in the segment fall below `MIN_SNAPSHOTS_PER_WINDOW`, exclude the object; otherwise, return {norad_id: {timestamps, features[T,6]}}.

**2. Variable-Length Sequence Padding & Masking**
- Since update frequencies vary per object (dense for continuously tracked LEO objects vs sparse for HEO/long-period objects), sequences have varying lengths (T_i). Instead of truncation or repetition, sequences are padded and masked to enable batch training without information loss or distortion.
- PyTorch Dataset + collate_fn + masked MSE loss

**3. Model Architecture & PyTorch Lightning Training Loop**
- LSTM Autoencoder Architecture: Encoder LSTM → Latent State (h_n[-1]) Extraction → Sequence Repeat → Decoder LSTM → Reconstruction
- The reconstruction loss serves directly as the orbital anomaly score (perturbation score).
- Padded timesteps are excluded from loss calculations during both training and evaluation using masked_mse_loss.

**4. Model Training & Registration**
- Input normal TLE sequences (e.g. continuous orbital trajectories over the past 30 days) into the LSTM Autoencoder to learn compressed representations and reconstructions of normal orbital perturbation patterns.
- Load processed Parquet files → Filter outdated TLEs → Construct sequences → Perform object-based 85/15 train/val split (preventing data leakage) → Train → Log to W&B
- To avoid consuming W&B artifact storage on the free tier, model checkpoints and scalers are uploaded to s3://models/, while logging only the corresponding S3 keys to W&B.
- The DAG handles training and checkpoint upload, leaving deployment to K8s rolling updates

### STEP 4. [추론/서빙] Inference & Serving (FastAPI)
- Training and inference share identical directory structures and feature extraction logic (sequence_builder and so on).
- FastAPI Serving: Accepts a target NORAD ID and returns its orbital anomaly score (reconstruction loss)
- If a satellite undergoes a sudden trajectory deviation or anomalous orbit due to collision risks, the model fails to reconstruct the pattern, causing reconstruction loss to spike. This loss value is directly used as the perturbation score.
- Automatically fetches and loads the latest checkpoint from S3 upon startup (no hot reloading)
- Upon request, aggregates recent processed snapshots from S3 (with caching enabled) and constructs target sequences using the exact same pipeline as training (stale TLE filter → windowing → feature extraction) for inference.

### [Interim Status] Docker Compose Services
- `ftor-ingestion`: Build-only container; used as a sibling container via DockerOperator.
- `ftor-model`: Build-only container; utilized by the ftor_model_training DAG.
- `model-training`: Executed automatically on a daily schedule via the ftor_model_training DAG (manual execution is reserved for local debugging/testing).
- `model-serving`: Always-on daemon listening on port 8000; isolated from Airflow orchestration.

### STEP 5. [배포/자동화] Local K8s Cluster & CI/CD (Minikube/GitHub Actions)
**1. Minikube**
- Configured the model serving environment on a local cluster using Deployment, Service, and HPA manifests.
- Used the /health endpoint for readiness, liveness, and startup probes.
- Directly references the locally built ftor-model:v1 image, running Uvicorn serving without pushing to a remote registry.
- Patching the Deployment annotation in CI triggers a rolling restart, updating to the latest model with zero downtime. (Verified via cluster logs showing the new Pod reaching Ready status prior to the old Pod entering Terminating)
- The Service exposes only the cluster-internal network via ClusterIP (external access via Ingress is planned for future phases).
- HPA (HorizontalPodAutoscaler) automatically scales replicas based on a 70% CPU target utilization.

**2. GitHub Actions**
- Since GitHub-hosted default runners cannot access the local Minikube kubectl context, registered a self-hosted runner directly on the host machine to maintain an always-on deployment agent.
- Cluster access privileges currently run on the local kubeconfig credentials of the host machine. Be aware that adding collaborators to the GitHub repository in the future poses security risks regarding potential leakage of Secrets or credentials via workflows, requiring careful access management.
- Added scheduled workflows in GitHub Actions to trigger automatically at designated daily times (scheduled post-model training), with support for manual workflow_dispatch execution.
- By architectural design, Airflow does not handle K8s deployments; instead, an independent script runs via GitHub Actions schedule to compare the latest checkpoint in S3 with the active Deployment. Upon detecting changes, it patches annotations to produce a Pod template diff, triggering a zero-downtime rolling restart without rebuilding images.
- Automatically rolls back to the previous revision if the rollout fails to succeed within the timeout window.

---

## **💡 Insights from Trial and Error**
> **PHASE 1:**
- **[STEP 1]** In the prototype, ingestion was limited to 100 recently updated objects within 3 days. Switched to full catalog ingestion for production readiness. Consequently, calculating pairwise distances across hundreds of millions of combinations became computationally infeasible via brute force. Adopted a two-stage approach similar to real-world operational systems (e.g. CelesTrak, SOCRATES): primary filtering by orbital similarity groups (altitude, inclination), followed by KDTree spatial indexing for fast proximity candidate extraction.

- **[STEP 2] Orbital Element Variations**
  - Evaluating rates of change for orbital elements (inclination, eccentricity, argument of pericenter, RAAN, mean motion, BSTAR) at 1-hour scheduling intervals revealed that 35,051 out of 35,052 objects shared identical epochs, as TLEs update only 2–4 times per day. Adjusted comparison logic to evaluate current snapshots against those from 24 hours prior.
  - Replaced the logic listing the full processed/ directory history on every run with listing only today's and yesterday's partitions. This prevents query overhead from escalating as dataset size grows.
  - Identified extreme spikes in `DELTA_MEAN_MOTION_PER_HR` (reaching hundreds of units). When dt_hours is excessively small (seconds to minutes), division becomes unstable, causing minor fluctuations to explode when converted to hourly rates. Introduced logic to set rate-of-change features to NaN when time intervals fall below a minimum threshold.

- **[STEP 2]** Cases where MIN_DISTANCE_KM == 0 (35 occurrences): Verified as expected behavior rather than a software bug. These represent physically attached structures evaluated near their respective TLE epochs without common-epoch propagation.<br>
  *Example:* ISS modules (NORAD IDs 25544, 25575, 26400, 26700, 36086: Zarya, Unity, Zvezda, Destiny, Poisk) carry separate NORAD IDs despite forming a single physical structure, resulting in identical spatial coordinates.

- **[STEP 3]** Detection of Future EPOCH Dates: Verified host system time integrity inside the container, then inspected raw JSON to confirm Space-Track explicitly serves future epochs. Objects exhibited `MEAN_MOTION` values below 1.0 rev/day, identifying them as high-altitude, long-period orbits (GTO, HEO, near-lunar). Radar tracking opportunities for such orbits are infrequent, making epochs shifted several days to two weeks into the future standard operational behavior. Since these are automatically filtered by min_snapshots conditions down the line, no manual correction is required.

- **[STEP 2]** Observed cases where newly launched satellites with unstable BSTAR estimates had anomaly scores dominated 100% by BSTAR alone. Removed `ORBITAL_DEVIATION_METRIC` (a single composite score summing element-wise deltas on fixed scales) from model inputs. Instead, raw element-wise deltas across 6 columns are directly fed into the model, allowing the LSTM Autoencoder to learn normal patterns autonomously.

- **[STEP 3]** As an LSTM Autoencoder relies on sequential time-series patterns, continuous ingestion of real raw data during development is essential to build sufficient dataset volume.

- **[STEP 3]** Excluded obsolete TLEs untouched for decades (e.g. VENERA 2 from 1965), as they fail to reflect current orbital states. Out of 2,733 objects filtered by the >30-day rule, a significant portion belonged to WESTFORD NEEDLES (tiny copper needle debris from 1960s military experiments). Considering pre-filtering these by name during TLE API ingestion, as tiny needle debris poses negligible physical impact risk to operational satellites.

- **[STEP 3]** LSTM Autoencoder val_loss Spikes: In near-circular orbits ($e \approx 0$), the physical concept of "pericenter" becomes ill-defined. Consequently, `ARG_OF_PERICENTER` fluctuates wildly across TLE refits regardless of physical trajectory changes. Updated preprocessing to set `ARG_OF_PERICENTER` deltas to NaN when eccentricity falls below a threshold (near-circular orbit).

- **[STEP 3]** Rationale for Robust Scaling (Median/IQR) over Mean/Std in sequence_builder: Features such as `DELTA_BSTAR_PER_HR` frequently exhibit extreme outliers, like in newly launched satellites. Mean and standard deviation are sensitive to these extremes (Caused by the same issue already identified in STEP 2). Median and IQR remain robust against extreme values, providing a stable baseline for "typical" orbital behavior.

- **[STEP 3]** Rationale for Percentile Clipping (1st/99th) in sequence_builder: Angular delta features such as `DELTA_ARG_OF_PERICENTER_PER_HR`, `DELTA_RA_OF_ASC_NODE_PER_HR` produce unphysical outliers when $0^\circ/360^\circ$ boundary wraparound is unhandled during preprocessing (e.g. $359.9^\circ → 0.1^\circ$ calculated as $-359.8^\circ$; observed normal range IQR ~0.2 vs observed max 67). Passing unclipped values after scaling causes MSE loss to be dominated by a small number of anomalies. Applied 1st–99th percentile clipping to ensure training stability.

- **[STEP 3]** Object-Based Data Splitting: Since sequence_builder currently generates "one latest window" per object, one object effectively corresponds to one sample. Time-based splitting would further fragment short single-object sequences arbitrarily. Splitting is therefore performed at the object level across train/val sets. Scalers are computed exclusively on raw DataFrames from training objects to prevent data leakage from validation sets.

> **PHASE 2:**
- **[STEP 5]** Evaluated GitOps (ArgoCD-based automation), Helm, and Kustomize during initial design, but deemed them over-engineered for a single model-serving pipeline. Excluded as they exceed the current scope and purpose of the project.

- **[STEP 5]** Placed S3_BUCKET_NAME in Kubernetes Secrets (injected into the cluster via CLI, excluded from version control) rather than ConfigMap (plaintext commits). Even for private buckets, exposed bucket names can incur costs from denied HTTP 403 request spikes. Additionally, set up a CloudWatch alarm to monitor sudden surges in S3 request volume.

- **[STEP 5]** Identified an issue where load_processed_snapshots read all parquet columns, loading unused metadata like redundant 69x2 TLE strings and ECI/ECEF/LLA coordinate frames into memory. Defined `REQUIRED_SNAPSHOT_COLS` to filter and load only essential inference columns (NORAD_CAT_ID, OBJECT_NAME, EPOCH, etc.), preventing recurrent OOMKilled crashes on the model-serving Pod. Added PYTHONUNBUFFERED=1 to ensure pre-crash stdout logs are flushed immediately without stdout buffering.

- **[STEP 5]** Initially configured the memory limit to 2Gi, which resulted in OOMKilled errors. Empirical measurements indicated steady-state consumption around ~2,821 MiB, with temporary memory spikes during cache TTL refreshes. Adjusted the memory limit to 3.5Gi to provide adequate headroom.

- **[STEP 5]** Encountered an issue immediately after deployment where readinessProbe failed continuously, preventing the Pod from transitioning to the Ready state. Liveness and readiness probes were initiating while serve.py was downloading checkpoints from S3, causing timeout-driven restart loops. Introduced a dedicated startupProbe to accommodate initial loading latencies before handing health checking over to liveness and readiness probes.

- **[STEP 5]** Configured maxUnavailable: 0 and maxSurge: 1, ensuring active Pods continue serving incoming traffic until new Pods achieve Ready status, enabling zero-downtime deployments even with replicas=1.

- **[STEP 5]** Initially, I used regular polling via crontab, but the GHA scheduler turned out to be like a batshit crazy FedEx driver who treats delivery times as mere suggestions, causing the deployment pipeline to get delayed by hours. Turns out GitHub's internal scheduler queue bottleneck is a chronic disease (they even admit in the official docs that punctuality isn't guaranteed), so I ditched the batch scheduling approach. Switched to an AWS Lambda-based event-driven push pipeline, so the deployment kicks off the exact second the model gets dumped into S3 (which is way more legit for MLOps automation anyway).<br>
Though, during the debugging process, I did all kinds of dumb bullshit to trace if it was a GitHub bug, changing the default branch, tweaking crontab to weird-ass minutes like :28 to dodge peak-hour bottlenecks, digging through mountain-high logs..<br>
TL;DR: Don't use GHA schedules for production.

---

## 📊 MLOps Pipeline & Workflow Execution
### 1. Experiment Logger
![wandb1](./assets/wandb1.png)

### 2. Actions & CI/CD Workflows
![workflow1](./assets/workflow1.png)
![workflow2](./assets/workflow2.png)

---

## **📜 Project Development Log**
#### 2026-08-19
- Project Kickoff: Inspired by Rocket Lab (after watching the HBO documentary "Wild Wild Space")
- Came up with a concept while listening to The Enid's debut album:<br>
  *"In the region of the summer stars💫, follow the white rabbit..🐇 into the orbital debris zone.✨"*
- Set up GitHub repository

#### 2026-08-20 ~ 2026-08-21
- Signed up for Space-Track.org
- Configured local development environment (WSL2, Docker, Airflow, etc.)
- Created AWS S3 bucket

#### 2026-08-22 ~ 2026-08-23
- Designed project architecture
- Authored ingestion scripts
- Ingested 100 recent objects for prototype pipeline testing
- Implemented Dockerfile and verified standalone execution

#### 2026-08-24 ~ 2026-08-25
- Started preprocessing script development
- Acquired domain knowledge to build rule-based algorithm logic for preprocessing
- Verified local execution via uv run airflow standalone
- Created docker-compose.yaml
- Configured scheduled test runs for hourly full-catalog ingestion
- Performed sample schema validation on mandatory fields against initial records

#### 2026-08-26 ~ 2026-08-27
- Conducted the 2nd Preprocessing & Feature Engineering development; debugged outputs against transformed data
- Deep Space & Lunar Orbit Filtering: Excluded objects with ALT_KM > 50,000km while maintaining GEO/HEO objects
- Identified issue where the BSTAR feature dominated 100% of the composite `ORBITAL_DEVIATION_METRIC` (observed in QIANFAN series)

#### 2026-08-28 ~ 2026-08-29
- Initiated ML model development: Fetched preprocessed parquet files from S3 for training pipelines
- W&B Setup: Configured automated logging for training runs, hyperparameters, and metrics via W&B, while decoupling artifact storage to S3
- Debugged val_loss spikes in LSTM Autoencoder: Identified 138 extreme outlier cases where eccentricity was $\le 0.0019$ (near-circular orbits)
- Configured CORS settings

#### 2026-08-30 ~ 2026-08-31
- Started model serving implementation: Loaded corresponding timestamped checkpoint and scaler pairs from S3
- Implemented automatic restart logic for model-serving when checkpoint files are absent in S3
- Authored model_training_dag: While data ingestion runs hourly, the sequence window (`SEQUENCE_WINDOW_HOURS`, default 72h) exhibits negligible input distribution shifts across intra-day retraining. Set training frequency to daily execution (03:00 UTC, post daily ingestion accumulation)
- Built MVP dashboard using Streamlit (migration to React planned for Phase 3; user-friendly UI/UX design pending)<br>
  Directly calls FastAPI (serve.py) endpoints: /health, /score/{norad_cat_id}, explicitly mapping 404 and 422 (insufficient snapshots) responses to separate error messages

#### 2026-09-01 ~ 2026-09-02
- Produced Streamlit dashboard demon video
- Done with the README for Phase 1
- Published Phase 1 pre-release
- Initiated Phase 2 development
- Completed Minikube installation, deployment, and HPA load testing

#### 2026-09-03 ~ 2026-09-04
- Excluded Airflow Triggerer
- Authored rollout automation scripts
- Registered self-hosted runner (maintaining an active state for GitHub Actions)
- Verified manual execution tests via workflow_dispatch

#### 2026-09-05 ~ 2026-09-08
- Tested crontab scheduling for Actions runner deployment
- Migrated execution trigger from GHA cron scheduling to AWS Lambda event-driven push
- Designed message queue (MSGQ) architecture
- Authored Phase 2 README documentation

---

## **⚙️ Components**
### Architecture
Under design..

### Directory
```
├── .github/                      # GitHub Actions CI/CD automation
│   └── workflows/
│       └── deploy-model.yml
├── .venv/...                     # (excluded from GitHub)
├── assets/...                    # README images
├── dags/                         # (excluded from GitHub)
│   ├── ingestion_dag.py          # DAG for Space-Track TLE ingestion & preprocessing
│   └── model_training_dag.py     # DAG for LSTM Autoencoder model training
├── dashboard/                    # temporary MVP UI consuming serve.py HTTP API
│   ├── requirements.txt          # dashboard dependencies
│   └── streamlit_app.py          # Streamlit app
├── data-prepare/
│   ├── Dockerfile                # container image for ingestion/preprocessing
│   ├── ingestion.py              # catalog ingestion, validation, storage
│   ├── preprocessing.py          # coordinate transformation, validation, orbital element variations, proximity screening
│   └── requirements.txt          # ingestion/preprocessing dependencies
├── k8s/                          # Kubernetes Manifest
│   ├── 00-namespace.yaml
│   ├── 01-configmap.yaml
│   ├── 02-secret.yaml.example
│   ├── 03-deployment.yaml
│   ├── 04-service.yaml
│   └── 05-hpa.yaml
├── model/                        # shared directory containing feature logic for training & serving
│   ├── Dockerfile                # container image for training/serving
│   ├── model.py                  # LSTM Autoencoder architecture definition
│   ├── requirements.txt          # training/serving dependencies
│   ├── sequence_builder.py       # per-object windowing, gap handling (excluded from GitHub)
│   ├── serve.py                  # FastAPI inference serving
│   ├── torch_dataset.py          # padding & masking
│   └── train.py                  # sequence construction, model training, W&B logging
├── scripts/                      # Deployment automation scripts
│   └── deploy/
│       ├── check_and_rollout.py
│       └── requirements.txt
├── .env                          # environment variables
├── .env.example                  # template for environment variables
├── .gitignore
├── docker-compose.yml            # (excluded from GitHub)
├── Dockerfile.airflow            # custom Airflow container image
├── pyproject.toml                # project configuration & dependencies
├── README_KR.md
├── README.md
└── uv.lock                       # dependency lock file
```

---

## **💁🏻‍♀️ Disclaimer**
Proprietary orbit propagation algorithms, fine-tuned risk model weights, etc., are masked for IP protection.<br>
The repository demonstrates the end-to-end MLOps infrastructure and pipeline functionality using mock evaluation modules.<br>
<br>

<p align="center"><b>Copyright © 2026 Mua💋無我 by Karyx💫. All Rights Reserved.</b></p>
