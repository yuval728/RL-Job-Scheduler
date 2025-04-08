from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import numpy as np
from stable_baselines3 import SAC
from job_env import JobSchedulingEnv
from scheduler import BaseScheduler

app = FastAPI()

# Load the model
env = JobSchedulingEnv(num_jobs=5)
model = SAC.load("js_model_SAC.zip", env=env)

class JobInput(BaseModel):
    jobs: List[List[float]]

class BaselineInput(JobInput):
    algorithm: Literal["fifo", "sjf", "edf", "priority"]

@app.post("/predict/")
def predict_full_rl_schedule(data: JobInput):
    all_jobs = np.array(data.jobs, dtype=np.float32)
    remaining_jobs = all_jobs.copy()
    env.reset(jobs=remaining_jobs.tolist())

    completed_jobs = []
    total_reward = 0

    while len(remaining_jobs) > 0:
        obs = np.array(remaining_jobs, dtype=np.float32)
        action_vector, _ = model.predict(obs)

        # Find most similar job (cosine similarity)
        similarities = [np.dot(action_vector, job) / (np.linalg.norm(action_vector) * np.linalg.norm(job) + 1e-8)
                        for job in remaining_jobs]
        selected_idx = int(np.argmax(similarities))
        selected_job = remaining_jobs[selected_idx]

        # Step in env with selected job
        obs_, reward, done, _, info = env.step(selected_idx)

        total_reward += reward
        completed_jobs.append((selected_job.tolist(), info["start_time"], info["end_time"]))

        # Remove selected job
        remaining_jobs = np.delete(remaining_jobs, selected_idx, axis=0)

    # Calculate average metrics
    waiting_times = [start - job[0] for job, start, _ in completed_jobs]
    turnaround_times = [end - job[0] for job, _, end in completed_jobs]
    avg_waiting_time = sum(waiting_times) / len(waiting_times)
    avg_turnaround_time = sum(turnaround_times) / len(turnaround_times)

    return {
        "avg_waiting_time": avg_waiting_time,
        "avg_turnaround_time": avg_turnaround_time,
        "total_reward": total_reward,
        "completed_jobs": completed_jobs
    }

@app.post("/baseline/")
def baseline_schedule(data: BaselineInput):
    scheduler = BaseScheduler(data.jobs)
    scheduler.reset()

    if data.algorithm == "fifo":
        result = scheduler.fifo()
    elif data.algorithm == "sjf":
        result = scheduler.sjf()
    elif data.algorithm == "edf":
        result = scheduler.edf()
    elif data.algorithm == "priority":
        result = scheduler.priority_based()
    else:
        return {"error": "Invalid algorithm"}

    return result
