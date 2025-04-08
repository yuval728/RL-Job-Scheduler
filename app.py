import streamlit as st
import requests
import numpy as np

st.title("Job Scheduling RL Agent 🧠📅")

num_jobs = st.slider("Number of Jobs", min_value=2, max_value=10, value=5)
st.write("Enter job features: (arrival_time, execution_time, priority, deadline, cpu_req, job_type)")

jobs = []
for i in range(num_jobs):
    with st.expander(f"Job {i + 1}"):
        arrival = st.slider(f"Arrival Time {i+1}", 0.0, 1.0, 0.2)
        exec_time = st.slider(f"Execution Time {i+1}", 0.1, 1.0, 0.5)
        priority = st.slider(f"Priority {i+1}", 0.2, 1.0, 0.6)
        deadline = st.slider(f"Deadline {i+1}", 0.2, 1.0, 0.8)
        cpu = st.slider(f"CPU Requirement {i+1}", 0.1, 1.0, 0.3)
        job_type = st.selectbox(f"Job Type {i+1}", [0, 1])
        jobs.append([arrival, exec_time, priority, deadline, cpu, float(job_type)])

if st.button("Predict Job Scheduling Decision"):
    with st.spinner("Calling RL agent..."):
        response = requests.post("http://localhost:8000/predict/", json={"jobs": jobs})
        if response.status_code == 200:
            result = response.json()
            st.success("Action predicted!")
            st.json(result)
        else:
            st.error("Something went wrong with the prediction.")
