import numpy as np

class BaseScheduler:
    def __init__(self, jobs):
        """
        jobs: List of jobs, where each job is a list of [arrival_time, execution_time, priority, deadline, cpu_req, job_type]
        """
        self.original_jobs = jobs
        self.jobs = [job.copy() for job in jobs]  # Avoid modifying the original list

    def reset(self):
        self.jobs = [job.copy() for job in self.original_jobs]

    def fifo(self):
        self.jobs.sort(key=lambda x: x[0])  # Sort by arrival_time
        return self._simulate()

    def sjf(self):
        self.jobs.sort(key=lambda x: x[1])  # Sort by execution_time
        return self._simulate()

    def edf(self):
        self.jobs.sort(key=lambda x: x[3])  # Sort by deadline
        return self._simulate()

    def priority_based(self):
        self.jobs.sort(key=lambda x: -x[2])  # Sort by priority (higher is better)
        return self._simulate()

    def _simulate(self):
        time = 0
        total_waiting_time = 0
        total_turnaround_time = 0
        completed_jobs = []

        for job in self.jobs:
            arrival, execution, priority, deadline, cpu, jtype = job
            if time < arrival:
                time = arrival
            waiting_time = time - arrival
            turnaround_time = waiting_time + execution
            time += execution
            total_waiting_time += waiting_time
            total_turnaround_time += turnaround_time
            completed_jobs.append((job, waiting_time, turnaround_time))

        avg_waiting_time = total_waiting_time / len(self.jobs)
        avg_turnaround_time = total_turnaround_time / len(self.jobs)

        return {
            "avg_waiting_time": avg_waiting_time,
            "avg_turnaround_time": avg_turnaround_time,
            "completed_jobs": completed_jobs
        }
