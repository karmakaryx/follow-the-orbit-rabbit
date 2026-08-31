import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # FastAPI 서버 주소
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

st.set_page_config(page_title="FTOR Collision Anomaly Dashboard", page_icon="🐇", layout="wide")


def fetch_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.sidebar.error(f"Health check failed: {e}")
        return None


def fetch_score(norad_cat_id: int) -> tuple[dict | None, str | None]:
    """단일 NORAD ID 조회. 성공 시 (payload, None), 실패 시 (None, error_message) 반환"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/score/{norad_cat_id}", timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    if resp.status_code == 200:
        return resp.json(), None
    if resp.status_code == 404:
        return None, f"NORAD_CAT_ID {norad_cat_id} not found in catalog"
    if resp.status_code == 422:
        try:
            detail = resp.json().get("detail", "Unprocessable entity")
        except ValueError:
            detail = "Unprocessable entity"
        return None, detail
    return None, f"Unexpected error (status {resp.status_code}): {resp.text[:200]}"


# ---------- 사이드바 ----------
st.sidebar.title("🐇 FTOR")
st.sidebar.caption("Follow The Orbit Rabbit: Collision Anomaly Scoring")
st.sidebar.text_input("API Base URL", value=API_BASE_URL, disabled=True)

health = st.sidebar.empty()
if st.sidebar.button("Health check", use_container_width=True):
    result = fetch_health()
    if result:
        health.success(f"status={result.get('status')} / model_loaded={result.get('model_loaded')}")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Temporary MVP dashboard. The official serving UI will be built with React.")

# ---------- 메인 ----------
st.title("🛰️ Collision Anomaly Scoring")
tab_single, tab_batch = st.tabs(["Single Query", "Batch Query (Watchlist)"])

# ----- 단일 조회 -----
with tab_single:
    col_input, col_button = st.columns([3, 1], vertical_alignment="bottom")
    with col_input:
        norad_id_input = st.number_input(
            "NORAD_CAT_ID", min_value=1, step=1, value=58800
        )
    with col_button:
        query_clicked = st.button("Search", type="primary", use_container_width=True)

    if query_clicked:
        with st.spinner(f"Scoring NORAD {int(norad_id_input)}..."):
            payload, error = fetch_score(int(norad_id_input))

        if error:
            st.error(error)
        else:
            score = payload["anomaly_score"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Anomaly Score (Reconstruction Loss)", f"{score:.6f}")
            col2.metric("Snapshots Used", payload["num_snapshots_used"])
            col3.metric("Latest Epoch (UTC)", payload["latest_epoch"][:19])

            st.write(f"**Object:** {payload['object_name']} (NORAD {payload['norad_cat_id']})")
            st.caption(
                "The score represents the reconstruction loss against the learned normal orbit behavior patterns. "
                "Higher values indicate a higher likelihood of orbital anomalies. "
                "Absolute risk thresholds have not been calibrated yet, so please use this score for relative comparison and monitoring purposes only."
            )
            st.caption(
                "점수는 학습된 정상 궤도 변화 패턴 대비 재구성 오차(reconstruction loss)이며, "
                "값이 높을수록 궤도 이상 가능성이 큽니다. 절대적인 위험도 임계치는 아직 캘리브레이션되지 않았으니 상대 비교/모니터링 용도로만 참고하세요."
            )

# ----- 복수 조회 -----
with tab_batch:
    col_batch_input, col_batch_button = st.columns([3, 1], vertical_alignment="bottom")
    with col_batch_input:
        ids_raw = st.text_area("NORAD_CAT_IDs", value="58800, 25544, 26400, 59474, 56708, 43000, 48274, 43226, 62916, 57990", height=80)
    with col_batch_button:
        batch_clicked = st.button("Batch Search", type="primary", use_container_width=True)

    st.caption("Enter multiple comma-separated NORAD IDs to view results sorted in descending order by anomaly score.")

    if batch_clicked:
        try:
            norad_ids = [int(x.strip()) for x in ids_raw.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid input: NORAD_CAT_ID list must be comma-separated integers")
            norad_ids = []

        if norad_ids:
            rows = []
            progress = st.progress(0.0)
            for i, nid in enumerate(norad_ids):
                payload, error = fetch_score(nid)
                if payload:
                    rows.append(
                        {
                            "NORAD_CAT_ID": payload["norad_cat_id"],
                            "OBJECT_NAME": payload["object_name"],
                            "ANOMALY_SCORE": payload["anomaly_score"],
                            "NUM_SNAPSHOTS_USED": payload["num_snapshots_used"],
                            "LATEST_EPOCH": payload["latest_epoch"],
                            "STATUS": "OK",
                        }
                    )
                else:
                    rows.append(
                        {
                            "NORAD_CAT_ID": nid,
                            "OBJECT_NAME": None,
                            "ANOMALY_SCORE": None,
                            "NUM_SNAPSHOTS_USED": None,
                            "LATEST_EPOCH": None,
                            "STATUS": error,
                        }
                    )
                progress.progress((i + 1) / len(norad_ids))

            df = pd.DataFrame(rows).sort_values(
                "ANOMALY_SCORE", ascending=False, na_position="last"
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            failed = df[df["STATUS"] != "OK"]
            if not failed.empty:
                st.warning(
                    f"{len(failed)} object(s) failed to score. See the STATUS column for details."
                )
