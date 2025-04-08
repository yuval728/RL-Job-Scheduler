from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import numpy as np
from stable_baselines3 import SAC
from job_env import JobSchedulingEnv
from scheduler import BaseScheduler

app = FastAPI()

# Load the environment and model
env = JobSchedulingEnv(num_jobs=5)
model = SAC.load("js_model_SAC.zip", env=env)


class JobInput(BaseModel):
    jobs: List[List[float]]


class BaselineInput(JobInput):
    algorithm: Literal["fifo", "sjf", "edf", "priority"]


@app.post("/predict/")
def predict_with_trained_model(data: JobInput):
    try:
        job_list = data.jobs
        env.reset(jobs=job_list)  # Set the environment with the given jobs

        obs, _ = env.reset()
        done = False
        rewards = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

        completed_jobs = env.done_jobs  # [(job, start_time, end_time), ...]

        # Metrics
        waiting_times = [start - job[0] for job, start, _ in completed_jobs]
        turnaround_times = [end - job[0] for job, _, end in completed_jobs]
        avg_waiting_time = sum(waiting_times) / len(waiting_times) + 1e-8
        avg_turnaround_time = sum(turnaround_times) / len(turnaround_times) + 1e-8

        return {
            "avg_waiting_time": float(avg_waiting_time),
            "avg_turnaround_time": float(avg_turnaround_time),
            "total_reward": float(np.sum(rewards)),
            "completed_jobs": [
                {
                    "job": [
                        float(x) for x in job
                    ],  # Convert each element to float for safe JSON encoding
                    "start_time": float(start),
                    "end_time": float(end),
                }
                for job, start, end in completed_jobs
            ],
        }

    except Exception as e:
        print(f"Error: {e}")


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
