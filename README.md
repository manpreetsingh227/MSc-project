# Bandit-based Approaches for Credit Risk Modeling under Drift and Selective Labels

- **Author:** Manpreet Singh  
- **Date:** 27/08/2025  

---

## Description

This repository contains the full pipeline for my MSc thesis.  
It evaluates **bandit-based algorithms** for credit risk modeling, focusing on challenges such as **distribution drift** and **selective labels**.  

The pipeline consists of preprocessing, feature selection, algorithm experiments (CF, OGD, RF), and analysis through notebooks.  

---
## 📂 Repository Structure

```text
msc-project/
├── src/
│   ├── merged.py             
│   ├── features.py         
│   ├── NEWCF.py           
│   ├── NEWOGD.py            
│   ├── NEWRF.py             
│
├── jobs/                     
│   ├── merged.pbs
│   ├── features.pbs
│   ├── cf.pbs
│   ├── ogd.pbs
│   ├── rf.pbs
│
├── notebooks/                
│   └── analysis.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```
---


## Workflow

1. **Preprocessing**:  
   Run `src/merged.py` to generate `notebooks/oldmerged.parquet`.  

2. **Feature selection**:  
   Run `src/features.py` to train CatBoost and compute feature importances.  
   - Output: `results/importance.csv`.  
   - Note: Algorithms use a hardcoded top-100 feature list; this file is for verification.  

3. **Algorithms**:  
   Run experiments using:  
   - `src/CF.py` (Cumulative refit)  
   - `src/OGD.py` (Online Gradient Descent)  
   - `src/RF.py` (Rolling refit)  
   Each supports modes (`oracle`, `bandit`, `epsilon_greedy`) and outputs CSV result files.  

4. **Analysis**:  
   Use `notebooks/analysis.ipynb` to plot metrics, compare algorithms, and generate figures for the thesis.  

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- catboost  
- pyarrow  
- joblib  
- jupyter *(optional, for notebooks)*  

Dependencies are listed in `requirements.txt`.  

---

## HPC Usage

Job submission scripts are provided in the `jobs/` directory.  
Each script corresponds to one Python script in `src/`.  

Examples:  
- `jobs/merged.pbs` → runs `src/merged.py`  
- `jobs/features.pbs` → runs `src/features.py`  
- `jobs/cf.pbs` → runs `src/CF.py`  
- `jobs/ogd.pbs` → runs `src/OGD.py`  
- `jobs/rf.pbs` → runs `src/RF.py`  

Submit jobs with:  
```bash
qsub jobs/<script>.pbs     # if cluster uses PBS
sbatch jobs/<script>.slurm # if cluster uses SLURM



