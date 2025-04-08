import streamlit as st
import requests
import pandas as pd
import random

st.set_page_config(page_title="Job Scheduling", layout="wide")
st.title("🧠📅 AI-Powered Job Scheduling")

num_jobs = 5
st.subheader("Enter Job Features")
st.caption("Each job: (id, arrival_time, execution_time, priority, deadline, cpu_req, job_type)")

# Initialize session state for jobs
if "jobs" not in st.session_state:
    st.session_state.jobs = [
        {
            "id": i + 1,
            "arrival_time": random.uniform(0.0, 1.0),
            "execution_time": random.uniform(0.1, 1.0),
            "priority": random.uniform(0.2, 1.0),
            "deadline": random.uniform(0.2, 1.0),
            "cpu_req": random.uniform(0.1, 1.0),
            "job_type": random.randint(0, 1),
        }
        for i in range(num_jobs)
    ]

# Display job inputs
for i, job in enumerate(st.session_state.jobs):
    with st.expander(f"Job {job['id']}", expanded=True):
        job["arrival_time"] = st.slider(
            f"Arrival Time {job['id']}", 0.0, 1.0, job["arrival_time"], key=f"arrival_{i}"
        )
        job["execution_time"] = st.slider(
            f"Execution Time {job['id']}", 0.1, 1.0, job["execution_time"], key=f"exec_{i}"
        )
        job["priority"] = st.slider(
            f"Priority {job['id']}", 0.2, 1.0, job["priority"], key=f"priority_{i}"
        )
        job["deadline"] = st.slider(
            f"Deadline {job['id']}", 0.2, 1.0, job["deadline"], key=f"deadline_{i}"
        )
        job["cpu_req"] = st.slider(
            f"CPU Requirement {job['id']}", 0.1, 1.0, job["cpu_req"], key=f"cpu_{i}"
        )
        job["job_type"] = st.selectbox(
            f"Job Type {job['id']}", [0, 1], index=job["job_type"], key=f"type_{i}"
        )

st.divider()

if st.button("🚀 Predict with RL & Baselines"):
    with st.spinner("Getting predictions..."):
        # Prepare jobs data for API
        jobs = [
            [
                job["arrival_time"],
                job["execution_time"],
                job["priority"],
                job["deadline"],
                job["cpu_req"],
                float(job["job_type"]),
            ]
            for job in st.session_state.jobs
        ]

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

            # Find the corresponding job id based on features
            matching_job = next(
                (job for job in st.session_state.jobs if
                job["arrival_time"] == features[0] and
                job["execution_time"] == features[1] and
                job["priority"] == features[2] and
                job["deadline"] == features[3] and
                job["cpu_req"] == features[4] and
                float(job["job_type"]) == features[5]),
                None
            )
            job_id = matching_job["id"] if matching_job else "Unknown"

            exec_time = features[1]
            waiting_time = start_time - features[0]
            turnaround_time = end_time - features[0]

            job_data.append([job_id] + features + [start_time, waiting_time, turnaround_time])
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)

        columns = ["ID", "Arrival", "Execution", "Priority", "Deadline", "CPU", "Type", "Start Time", "Waiting Time", "Turnaround Time"]
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

            # Find the corresponding job id based on features
            matching_job = next(
                (job for job in st.session_state.jobs if
                job["arrival_time"] == features[0] and
                job["execution_time"] == features[1] and
                job["priority"] == features[2] and
                job["deadline"] == features[3] and
                job["cpu_req"] == features[4] and
                float(job["job_type"]) == features[5]),
                None
            )
            job_id = matching_job["id"] if matching_job else "Unknown"

            exec_time = features[1]
            waiting_time = start_time - features[0]
            turnaround_time = end_time - features[0]

            job_data.append([job_id] + features + [start_time, waiting_time, turnaround_time])
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)

        columns = ["ID", "Arrival", "Execution", "Priority", "Deadline", "CPU", "Type", "Start Time", "Waiting Time", "Turnaround Time"]
        df = pd.DataFrame(job_data, columns=columns)

        st.dataframe(
            df.style.highlight_max(axis=0, subset=["Waiting Time", "Turnaround Time"], color="lightgreen"),
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        col1.metric("📉 Avg Waiting Time", f"{result['avg_waiting_time']:.3f}")
        col2.metric("⏱️ Avg Turnaround Time", f"{result['avg_turnaround_time']:.3f}")
        st.divider()

