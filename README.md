# AI-Driven Carbon-Aware Cloud Task Scheduler Using Deep Reinforcement Learning

## Overview
This project presents an AI-driven carbon-aware cloud task scheduling system using Deep Reinforcement Learning (DRL). The system optimizes task allocation in heterogeneous cloud data centers while minimizing carbon emissions, energy consumption, and SLA violations.

The project uses CloudSim Plus for workload simulation and a Deep Q-Network (DQN) agent for intelligent scheduling decisions based on real workload traces and carbon intensity data.

---

## Features
- Deep Reinforcement Learning–based cloud scheduling
- Carbon-aware task allocation
- Energy-efficient workload optimization
- SLA-aware scheduling strategy
- Real workload trace simulation using CloudSim Plus
- Heterogeneous server environment
- Comparative evaluation against traditional schedulers
- Performance metrics visualization

---

## Tech Stack
- Python
- Deep Q-Network (DQN)
- Stable-Baselines3
- OpenAI Gymnasium
- CloudSim Plus
- NumPy
- Matplotlib
- Java

---

## System Workflow
1. Generate cloud workloads using CloudSim Plus
2. Create heterogeneous cloud server environment
3. Collect workload and carbon intensity data
4. Train DQN agent for optimal task scheduling
5. Allocate tasks dynamically based on reward optimization
6. Evaluate energy usage, CO₂ emissions, and SLA performance

---

## Key Highlights
- Optimized scheduling across 20 servers and 1000+ workload tasks
- Reduced energy consumption and carbon emissions
- Improved SLA performance using reinforcement learning
- Simulated real-world cloud computing scenarios

---

## Project Structure
```text
carbon-aware-cloud-scheduler-drl/
│
├── src/
├── workloads/
├── metrics/
├── screenshots/
├── report/
├── requirements.txt
├── README.md
└── .gitignore

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Project
python train.py

---

## Results
The DRL-based scheduler demonstrated efficient workload allocation and improved carbon-aware scheduling performance compared to traditional scheduling approaches.

---

## Future Enhancements
Multi-cloud optimization
Real-time cloud deployment
Kubernetes integration
Advanced RL algorithms (PPO/A3C)
Dynamic renewable energy integration

---

## Contributors
Collaborative academic research project developed as part of cloud computing and AI-driven sustainability research.

## Author

Unnati Lunawat