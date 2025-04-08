import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Job Scheduling", layout="wide")
st.title("🧠📅 AI-Powered Job Scheduling")

num_jobs = 5
st.subheader("Enter Job Features")
st.caption("Each job: (arrival_time, execution_time, priority, deadline, cpu_req, job_type)")

jobs = []
for i in range(num_jobs):
    with st.expander(f"Job {i + 1}", expanded=True):
        arrival = st.slider(f"Arrival Time {i+1}", 0.0, 1.0, random.uniform(0.0, 1.0), key=f"arrival_{i}")
        exec_time = st.slider(f"Execution Time {i+1}", 0.1, 1.0, random.uniform(0.1, 1.0), key=f"exec_{i}")
        priority = st.slider(f"Priority {i+1}", 0.2, 1.0, random.uniform(0.2, 1.0), key=f"priority_{i}")
        deadline = st.slider(f"Deadline {i+1}", 0.2, 1.0, random.uniform(0.2, 1.0), key=f"deadline_{i}")
        cpu = st.slider(f"CPU Requirement {i+1}", 0.1, 1.0, random.uniform(0.1, 1.0), key=f"cpu_{i}")
        job_type = st.selectbox(f"Job Type {i+1}", [0, 1], index=random.randint(0, 1), key=f"type_{i}")
        jobs.append([arrival, exec_time, priority, deadline, cpu, float(job_type)])

st.divider()

if st.button("🚀 Predict with RL & Baselines"):
    with st.spinner("Getting predictions..."):

        # --- RL Prediction ---
        rl_response = requests.post("http://localhost:8000/predict/", json={"jobs": jobs})
        rl_result = rl_response.json() if rl_response.status_code == 200 else None

        # --- Baseline Predictions ---
        baseline_results = {}
        for algo in ["fifo", "sjf", "edf", "priority"]:
            res = requests.post("http://localhost:8000/baseline/", json={"jobs": jobs, "algorithm": algo})
            if res.status_code == 200:
                baseline_results[algo.upper()] = res.json()

    st.success("✅ Predictions Ready")

    # ===============================
    # RL AGENT PREDICTION INTERPRETATION
    # ===============================
    if rl_result:
        st.subheader("🧠 RL Agent Schedule")

        job_data = []
        waiting_times = []
        turnaround_times = []

        for job_entry in rl_result["completed_jobs"]:
            features = job_entry["job"]
            start_time = job_entry["start_time"]
            end_time = job_entry["end_time"]


            exec_time = features[1]
            waiting_time = start_time - features[0]
            turnaround_time = end_time - features[0]

            job_data.append(features + [start_time, waiting_time, turnaround_time])
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)

        columns = ["Arrival", "Execution", "Priority", "Deadline", "CPU", "Type", "Start Time", "Waiting Time", "Turnaround Time"]
        df = pd.DataFrame(job_data, columns=columns)

        st.dataframe(
            df.style.highlight_max(axis=0, subset=["Waiting Time", "Turnaround Time"], color="lightblue"),
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        col1.metric("📉 Avg Waiting Time", f"{rl_result['avg_waiting_time']:.3f}")
        col2.metric("⏱️ Avg Turnaround Time", f"{rl_result['avg_turnaround_time']:.3f}")
        st.metric("💰 Total Reward", f"{rl_result['total_reward']:.3f}")
        st.divider()

    # ===============================
    # BASELINE RESULTS
    # ===============================
    st.subheader("📊 Baseline Algorithm Results")
    for algo, result in baseline_results.items():
        st.markdown(f"### 🔹 {algo} Scheduling")

        # Extract job info from completed_jobs
        job_data = []
        waiting_times = []
        turnaround_times = []

        for job in result["completed_jobs"]:
            features, start_time, end_time = job
            exec_time = features[1]
            waiting_time = start_time - features[0]
            turnaround_time = end_time - features[0]

            job_data.append(features + [start_time, waiting_time, turnaround_time])
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)

        columns = ["Arrival", "Execution", "Priority", "Deadline", "CPU", "Type", "Start Time", "Waiting Time", "Turnaround Time"]
        df = pd.DataFrame(job_data, columns=columns)

        st.dataframe(
            df.style.highlight_max(axis=0, subset=["Waiting Time", "Turnaround Time"], color="lightgreen"),
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        col1.metric("📉 Avg Waiting Time", f"{result['avg_waiting_time']:.3f}")
        col2.metric("⏱️ Avg Turnaround Time", f"{result['avg_turnaround_time']:.3f}")
        st.divider()
