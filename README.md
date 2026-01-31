# Traffic Noise Prediction Using Artificial Neural Networks

**Tehran Urban Roads – Reproducible Case Study**

## 🔍 Problem Context

Accurate prediction of urban traffic noise is critical for environmental impact assessment, urban planning, and road infrastructure management. Traditional empirical and regression-based noise models struggle to capture nonlinear relationships between traffic, geometry, and surrounding built environments.

## 🎯 Objective

This repository provides a **reproducible Python implementation** of the study:

> *Mansourkhaki et al. (2018), “A neural network noise prediction model for Tehran urban roads”*

The goal is to:

* Implement an **Artificial Neural Network (ANN)** for predicting equivalent continuous sound level ($L_{Aeq}$)
* Compare ANN performance against a **Multiple Linear Regression (MLR)** baseline
* Demonstrate the impact of a **novel domain-specific parameter** (Building Reflection Factor)

> ⚠️ Note: Due to data availability constraints, this repository uses **synthetic data** generated to match the statistical characteristics and methodology described in the original paper.

---

## 🧠 Methodology

### Models Implemented

* **Multilayer Perceptron (MLP)** – primary model
* **Multiple Linear Regression (MLR)** – baseline comparator

### Neural Network Architecture

* Feed-forward ANN with **6–10–1** structure

  * 6 input features
  * 10 hidden neurons
  * 1 output ($L_{Aeq}$)

### Optimization Strategy

* Uses **LBFGS (Quasi-Newton)** optimizer
* Selected as a practical approximation of the **Levenberg–Marquardt** algorithm used in the original study

---

## 📊 Input Features

Based on the optimal configuration (Scenario 4) reported in the paper:

1. **Q** – Traffic volume (veh/h)
2. **V** – Average vehicle speed
3. **PH** – Percentage of heavy vehicles
4. **G** – Road gradient
5. **D** – Building density around the road
6. **BRF** – *Building Reflection Factor* (novel parameter)

### 🔹 Building Reflection Factor (BRF)

The BRF captures the effect of sound reflections from building facades adjacent to urban road corridors — a key contribution of the original study and an important extension beyond classical noise models.

---

## 🧪 Data Splitting Strategy

To remain consistent with the original methodology, data is randomly split into:

* **80%** Training
* **10%** Validation
* **10%** Testing

---

## 🚀 Results (Expected Behavior)

Consistent with the published study, the ANN model:

* Outperforms multiple linear regression
* Captures nonlinear relationships between traffic, geometry, and built environment
* Demonstrates the relevance of domain-informed input features (e.g., BRF)

> Exact numerical performance depends on synthetic data realization but follows the reported trends in the original paper.

---

## ▶️ How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train and evaluate the model**

```bash
python train_mlp.py
```

---

## 🏗️ Engineering & Research Relevance

This work demonstrates how machine learning can:

* Enhance **environmental noise assessment**
* Support **urban road design and policy evaluation**
* Replace or augment empirical noise prediction models with data-driven approaches

The repository serves as a **reference implementation** for applying ANN techniques in transportation and environmental engineering contexts.

---

## 📄 Reference

Mansourkhaki, A., **Berangi, M.**, Haghiri, M., & Haghani, M. (2018).
*A neural network noise prediction model for Tehran urban roads.*
**Journal of Environmental Engineering and Landscape Management**, 26(2), 88–97.
[https://doi.org/10.3846/16486897.2017.1356327](https://doi.org/10.3846/16486897.2017.1356327)
