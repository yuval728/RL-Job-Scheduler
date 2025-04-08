from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from stable_baselines3 import SAC  # or PPO/A2C
from job_env import JobSchedulingEnv  # Your custom environment

app = FastAPI()

# Load the model
env = JobSchedulingEnv(num_jobs=10)
model = SAC.load("js_model_SAC.zip", env=env)  # Change path/algorithm as needed

class JobInput(BaseModel):
    jobs: List[List[float]]  # shape (num_jobs, 6)

@app.post("/predict/")
def predict_action(data: JobInput):
    obs = np.array(data.jobs, dtype=np.float32)
    action, _ = model.predict(obs)
    obs_, reward, done, _, _ = env.step(action)
    return {
        "action": action.tolist(),
        "reward": reward,
        "done": done
    }
