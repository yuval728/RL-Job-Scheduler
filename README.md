# AI-Powered Job Scheduling System

This project implements an **AI-powered job scheduling system** that combines **Reinforcement Learning (RL)** and traditional scheduling algorithms to optimize job scheduling. The system is designed to predict job schedules, evaluate performance metrics, and compare RL-based scheduling with baseline algorithms.

---

## Features

1. **Reinforcement Learning (RL) Scheduling**:

   - Uses the **Soft Actor-Critic (SAC)** algorithm to train an RL agent for job scheduling.
   - Optimizes metrics such as waiting time, turnaround time, and total reward.

2. **Baseline Scheduling Algorithms**:

   - Implements traditional scheduling algorithms:
     - **FIFO**: First-In-First-Out.
     - **SJF**: Shortest Job First.
     - **EDF**: Earliest Deadline First.
     - **Priority-based**: Based on job priority.

3. **FastAPI Backend**:

   - Provides REST API endpoints for RL-based and baseline scheduling:
     - `/predict/`: Predicts schedules using the RL model.
     - `/baseline/`: Computes schedules using baseline algorithms.

4. **Streamlit Frontend**:

   - Interactive UI for users to input job details and view scheduling results.
   - Displays metrics and comparisons between RL and baseline algorithms.

5. **Visualization**:
   - Plots learning curves and performance metrics for RL training.
   - Highlights key metrics such as waiting time and turnaround time.

---

### App

Reinforcement Learning
Environment
The custom RL environment (JobSchedulingEnv) simulates the job scheduling problem. It includes:

Observation Space: Job features (arrival time, execution time, priority, deadline, CPU requirements, job type).
Action Space: Scheduling decisions and resource allocation.
Training
The RL agent is trained using the Soft Actor-Critic (SAC) algorithm. Training scripts and evaluation are implemented in job_scheduler.ipynb.

Metrics
Waiting Time: Time a job waits before execution.
Turnaround Time: Total time from job arrival to completion.
Total Reward: Cumulative reward obtained by the RL agent.
Baseline Algorithms
FIFO: Jobs are executed in the order of arrival.
SJF: Jobs with the shortest execution time are executed first.
EDF: Jobs with the earliest deadline are executed first.
Priority-based: Jobs with the highest priority are executed first.
Results
RL vs Baseline Comparison
RL-based scheduling optimizes metrics such as waiting time and turnaround time.
Baseline algorithms provide a reference for evaluating RL performance.
Visualization
Learning curves and reward trends are plotted during training.
Results are displayed in the Streamlit UI for easy comparison.

### Notebook

Reinforcement Learning for Job Scheduling
This project uses Reinforcement Learning (RL) to optimize job scheduling. The RL agent interacts with a custom environment (JobSchedulingEnv) to learn scheduling strategies that minimize waiting time, turnaround time, and energy consumption while maximizing throughput and rewards.

Custom Environment: JobSchedulingEnv
The JobSchedulingEnv class is a custom environment built using the Gymnasium framework. It simulates the job scheduling problem and provides the following features:

Environment Details
Observation Space: A matrix of shape (num_jobs, 6) where each row represents a job with the following attributes:

Arrival Time: When the job arrives in the system.
Execution Time: How long the job takes to execute.
Priority: The priority level of the job (higher is better).
Deadline: The time by which the job must be completed.
CPU Requirements: The computational resources required by the job.
Job Type: A binary value indicating the type of job.
Action Space: A vector of shape (num_jobs + 1,):

The first num_jobs elements represent scheduling scores for each job.
The last element represents resource allocation for the selected job.
Reset Method
Initializes the environment by generating random jobs or using user-provided jobs.
Jobs are sorted by arrival time for processing.
Step Method
Takes an action (scheduling scores and resource allocation).
Selects the job with the highest scheduling score.
Calculates the reward based on:
Waiting Time: Penalizes delays in starting the job.
Execution Time: Penalizes longer execution times.
Deadline Misses: Heavily penalizes jobs that miss their deadlines.
Energy Efficiency: Penalizes deviations from optimal resource allocation.
Throughput: Rewards faster job completion.
Updates the environment state and returns the next observation, reward, and termination status.
Baseline Scheduling Algorithms
The project includes traditional scheduling algorithms for comparison with RL-based scheduling:

FIFO (First-In-First-Out):Jobs are executed in the order of their arrival times.
SJF (Shortest Job First):Jobs with the shortest execution times are executed first.
EDF (Earliest Deadline First):Jobs with the earliest deadlines are executed first.
Reinforcement Learning Algorithms
The project evaluates multiple RL algorithms from Stable-Baselines3 to determine the best-performing model for job scheduling:

PPO (Proximal Policy Optimization)
A2C (Advantage Actor-Critic)
SAC (Soft Actor-Critic)
TD3 (Twin Delayed Deep Deterministic Policy Gradient)
DDPG (Deep Deterministic Policy Gradient)
Training Process
Each algorithm is trained on the JobSchedulingEnv environment for a fixed number of timesteps.
The training process uses a random seed for reproducibility.
TensorBoard logs are generated for monitoring training progress.
Evaluation
After training, each model is evaluated using:
Total Reward: The cumulative reward obtained during training.
Evaluation Reward: The average reward over multiple test episodes.
Results
The performance of each algorithm is compared using a bar chart of total rewards.
Soft Actor-Critic (SAC) Training
The Soft Actor-Critic (SAC) algorithm is selected as the primary RL model for job scheduling due to its superior performance. The training process includes:

Training:

The SAC model is trained for 20,000 timesteps.
Rewards are tracked per episode to monitor learning progress.
Testing:

The trained model is tested over 10 episodes to evaluate its performance.
Metrics such as average reward and the number of completed jobs are recorded.
Learning Curve:

A plot of rewards per episode is generated to visualize the agent's learning progress.
Comparison with Baselines
The RL agent's performance is compared with baseline algorithms (FIFO, SJF, EDF) using the following metrics:

Reward Over Time: A plot showing the RL agent's reward per step compared to baseline rewards.
Waiting Time: The total waiting time for all jobs.
Turnaround Time: The total time from job arrival to completion.
Visualization
The project includes several visualizations to analyze the RL agent's performance:

Bar Chart: Compares total rewards for different RL algorithms.
Learning Curve: Plots rewards per episode during training.
Reward Comparison: Shows the RL agent's reward over time compared to baseline algorithms.
Model Saving and Loading
The trained SAC model is saved to a file (js_model_SAC.zip) for future use.
The saved model can be loaded and tested on new job datasets.
Example Usage:
Key Results
RL Agent Performance:

The SAC agent achieves higher rewards compared to baseline algorithms.
It effectively balances waiting time, turnaround time, and energy efficiency.
Baseline Comparison:

FIFO and SJF are simple but fail to optimize for deadlines and priorities.
EDF performs better for deadline-sensitive jobs but lacks flexibility.
Visualization Insights:

The RL agent consistently outperforms baselines in terms of cumulative rewards.
This section provides a comprehensive explanation of the RL-based job scheduling system, its environment, algorithms, and results. Let me know if you need further refinements!
