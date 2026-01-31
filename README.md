# Traffic Noise Prediction using ANN (Tehran Case Study)

## Overview
This repository contains a  simple Python implementation of the research paper **"A Neural Network Noise Prediction Model for Tehran Urban Roads 2015"**  with a synthetic data. >> [link text](https://scholar.google.com/citations?user=k6hDU6sAAAAJ&hl=en).

The project predicts the equivalent sound level ($L_{Aeq}$) based on traffic and environmental characteristics using a Multilayer Perceptron (MLP) Artificial Neural Network. It compares the ANN performance against a Multiple Linear Regression model.

## Key Features
**Novel Parameter:** Implementation includes the **Building Reflection Factor (BRF)**, a parameter introduced in the study to account for noise reflected by building facades.

**Model Architecture:** A Feed-Forward Neural Network with a 6-10-1 architecture (6 inputs, 10 hidden neurons, 1 output).

**Uses the LBFGS solver (Quasi-Newton) to approximate the Levenberg-Marquardt** algorithm used in the original paper.

## Input Parameters
Based on the optimal scenario (Scenario 4) described in the paper:
1.  **Q**: Total traffic volume per hour.
2.  **V**: Average speed of vehicles.
3.  **PH**: Percentage of heavy vehicles.
4.  **G**: Road gradient.
5.  **D**: Density of buildings around the road section.
6.  **BRF**: Building Reflection Factor.

## Data Split
The data is split randomly as follows, matching the paper's methodology:
* **80%** Training
* **10%** Validation
* **10%** Testing

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

## Citation
> Mansourkhaki, A., Berangi, M., Haghiri, M., & Haghani, M. (2018). A neural network noise prediction model for Tehran urban roads. Journal of Environmental Engineering and Landscape Management, 26(2), 88–97. https://doi.org/10.3846/16486897.2017.1356327
