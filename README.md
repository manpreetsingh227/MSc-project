# Bandit-based approaches for Credit Risk Modeling under Drift and Selective Labels

- **Author:** Manpreet Singh  
- **Date:** 27/08/2025  

---

## 📖 Description

This repository contains all of the code for my MSc thesis.  
The primary goal of my research is to investigate **bandit-based algorithms** for credit risk modeling, with a focus on challenges such as **distribution drift** and **selective labels**.  

The pipeline consists of:
1. **Preprocessing** the raw credit risk dataset into a structured format.  
2. **Feature selection** using CatBoost feature importance.  
3. Implementing and evaluating **three algorithms**:  
   - Cumulative refit
   - Online Gradient Descent 
   - Rolling Fit   
4. **Analysis and visualization** of results through Jupyter notebooks.  

---

## 📂 File Descriptions

- `src/merged.py` : Preprocessing script to generate `oldmerged.parquet`.  
- `src/features.py` : Runs CatBoost to produce feature importances.  
- `src/CF.py` : Cumulative refit algorithm implementation.  
- `src/OGD.py` : Online Gradient Descent algorithm implementation.  
- `src/RF.py` : Rolling Fit algorithm implementation.  
- `notebooks/analysis.ipynb` : Notebook for plotting and comparing results.  
- `requirements.txt` : Python dependencies.  
- `README.md` : This file.  

---

## 📊 Data

- Raw parquet files are stored in the `data/` directory (**not tracked in GitHub**).  
- Preprocessing produces `oldmerged.parquet`, used as input for algorithms.  

---


## ⚙️ Workflow

### 1. Preprocessing  
Run `src/merged.py` to build the merged dataset.  
- Input: raw parquet files in `data/`.  
- Output: `data/oldmerged.parquet`.  

### 2. Feature Selection  
Run `src/features.py` to train CatBoost and compute importances.  
- Input: `data/oldmerged.parquet`.  
- Output: `results/importance.csv`.  
- Note: Algorithms use a hard-coded top 100 feature list; `importance.csv` is for verification.  

### 3. Algorithms  
Run the three sequential decision algorithms:  
- **Counterfactual (CF)** – `src/NEWCF.py`  
- **Online Gradient Descent (OGD)** – `src/NEWOGD.py`  
- **Rolling Fit (RF)** – `src/NEWRF.py`  

Each supports modes: `oracle`, `bandit`, `epsilon_greedy`.  
Outputs are CSV files such as:  
- `cf_oracle_seed42.csv`  
- `ogd_epsilon_greedy_eps0.1_seed123.csv`  
- `rf_bandit_seed123.csv`  

### 4. Analysis  
Open `notebooks/analysis.ipynb` to:  
- Plot utility, AUC, accept rates by segment.  
- Compare CF, OGD, RF across modes.  
- Generate figures for the MSc report.  

---

## ⚙️ Dependencies

- pandas  
- numpy  
- scikit-learn  
- catboost  
- pyarrow  
- joblib  
- jupyter *(optional for notebooks)*  

Dependencies are listed in `requirements.txt`.  

---

## 📌 Notes

- `data/` and `results/` are gitignored to keep the repo lightweight.  
- HPC job scripts live in `jobs/`.  
- Logs from jobs are stored in `logs/`.  
- Only source code, notebooks, and configs are tracked in Git.  

---

## 🖥️ HPC Usage

Job submission scripts are provided in `jobs/`. Each corresponds to one Python script in `src/`.  

### PBS Example
Submit preprocessing:
```bash
qsub jobs/merged.pbs


