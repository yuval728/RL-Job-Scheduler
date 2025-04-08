import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class JobSchedulingEnv(gym.Env):
    def __init__(self, num_jobs=5):
        super(JobSchedulingEnv, self).__init__()
        self.num_jobs = num_jobs
        
        # Each job has 6 features:
        # (arrival_time, execution_time, priority, deadline, cpu_requirement, job_type)
        self.jobs = []
        self.backup_jobs = []
        
        self.action_space = spaces.Box(low=0, high=1, shape=(num_jobs + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=100, shape=(num_jobs, 6), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        arrival_times = np.sort(np.random.normal(loc=5, scale=2, size=self.num_jobs))
        arrival_times = np.clip(arrival_times, 0, 10) / 10

        execution_times = np.clip(np.random.normal(loc=5, scale=2, size=self.num_jobs), 1, 10) / 10
        priorities = np.random.choice([1, 2, 3, 4, 5], self.num_jobs, p=[0.1, 0.2, 0.4, 0.2, 0.1]) / 5
        deadlines = (arrival_times + execution_times + np.random.randint(3, 10, size=self.num_jobs)) / 20
        cpu_requirements = np.random.randint(1, 11, size=self.num_jobs) / 10
        job_types = np.random.choice([0, 1], self.num_jobs)

        self.jobs = list(zip(arrival_times, execution_times, priorities, deadlines, cpu_requirements, job_types))
        self.jobs.sort(key=lambda x: x[0])
        self.backup_jobs = self.jobs.copy()
        self.time = 0
        self.done_jobs = []
        self.total_waiting_time = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        obs = np.zeros((self.num_jobs, 6), dtype=np.float32)
        for i, job in enumerate(self.jobs):
            obs[i] = job
        return obs

    def step(self, action):
        scheduling_scores = action[:self.num_jobs]
        resource_allocation = action[-1]
        
        job_index = int(np.argmax(scheduling_scores))
        
        if job_index >= len(self.jobs):
            return self._get_observation(), -10, True, False, {}
        
        selected_job = self.jobs.pop(job_index)
        arrival_time, execution_time, priority, deadline, cpu_req, job_type = selected_job
        
        self.time = max(self.time, arrival_time) + execution_time
        
        waiting_time = self.time - arrival_time - execution_time
        self.total_waiting_time += waiting_time
        base_reward = -waiting_time - (execution_time / priority) - len(self.jobs) * 0.1
        
        if self.time > deadline:
            base_reward -= 10
        else:
            base_reward += 5
        
        optimal_resource = 0.7
        energy_penalty = 5 * abs(resource_allocation - optimal_resource)
        throughput_reward = 10 / (self.time + 1)
        
        reward = base_reward + throughput_reward - energy_penalty
        
        self.done_jobs.append(selected_job)
        done = len(self.jobs) == 0
        
        return self._get_observation(), reward, done, False, {}
