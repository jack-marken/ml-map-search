
# Map Search With Trained Machine Learning Models - COS30019

This repository contains a traffic navigation system for the city of Boroondara, Victoria, informed by six search algorithms and three machine learning models.

Search algorithms:
- DFS
- BFS
- GBFS
- A*
- Iterative-depth DFS
- Beam Search

ML models:
- LSTM
- GRU
- RNN

The ML models were trained on real intersection traffic data collected by VicRoads, specifically in the month of October 2006 in the city of Boroondara. The program is designed to predict the fastest 5 paths from one origin SCATS site to one destination SCATS site, at a specified time and date within the month of November 2006.

# Installation - With Anaconda

```
> conda create -n trafficenv python=3.12.9
> conda activate trafficenv
> pip install -r requirements.txt
```

# Usage instructions
## Main project

```
> python main.py
```
