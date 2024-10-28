# Cybersecurity Detection Using Large Language Models (LLMs) on Tabular Data

This project explores the application of Large Language Models (LLMs) in the detection of cyberattacks within industrial environments, specifically focusing on tabular data from the Secure Water Treatment (SWaT) dataset. Two state-of-the-art LLMs, Falcon-7B and LLaMA2-7B, are utilized to evaluate their ability to detect anomalies and potential intrusions in a water treatment plant scenario. The research primarily leverages a zero-shot inference approach to test the generalization capabilities of these models in a cybersecurity context.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Models Used](#models-used)
4. [Zero-Shot Approach](#zero-shot-approach)
5. [Experiment Configuration](#experiment-configuration)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results and Insights](#results-and-insights)

---

### 1. Overview <a name="overview"></a>

This study investigates the performance of LLMs in detecting cyberattacks on industrial control systems, using the SWaT dataset as a testing environment. The dataset, which records both normal and attack simulations in a water treatment plant, allows for comprehensive anomaly detection testing. The models are tested for their capacity to identify malicious patterns within tabular time-series data without prior task-specific training.

### 2. Dataset <a name="dataset"></a>

The [SWaT (Secure Water Treatment)](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) dataset is widely used in cybersecurity research for evaluating intrusion detection in industrial systems. It contains operational data from a real water treatment plant, simulating various attack scenarios under normal operations. This time-series dataset includes sensor readings and actuator statuses, with labeled normal and malicious events.

### 3. Models Used <a name="models-used"></a>

The experiments were conducted using two prominent LLMs:
- **Falcon-7B**: A highly efficient open-source language model designed for a broad range of machine learning tasks.
- **LLaMA2-7B**: The second generation of the LLaMA model, optimized for language processing tasks, particularly zero-shot inference scenarios.

These models were chosen for their advanced generalization capabilities and relevance in current AI research.

### 4. Zero-Shot Approach <a name="zero-shot-approach"></a>

This study employs a zero-shot inference approach, where no specific training examples are provided to the models. Instead, a detailed dataset description, including information on sensors, actuators, and attack types, was provided to help the models infer patterns without explicit training on the detection task.

### 5. Experiment Configuration <a name="experiment-configuration"></a>

The models were set up to analyze sensor readings and actuator data, classifying each sequence of events as either normal or malicious. The experiments aimed to evaluate the models' abilities to:
1. Detect anomalous patterns in tabular data without explicit training;
2. Accurately classify malicious events, including denial-of-service attacks and sensor manipulations;
3. Maintain inference consistency across varying temporal segments of the dataset.

### 6. Evaluation Metrics <a name="evaluation-metrics"></a>

Performance evaluation was based on commonly used classification metrics:
- **Accuracy**: Proportion of correctly identified events (both normal and malicious).
- **Recall**: Ability to correctly detect malicious events.
- **F1-Score**: Harmonic mean of precision and recall, highlighting the balance between true positives and false positives.

### 7. Results and Insights <a name="results-and-insights"></a>

The findings from this study shed light on the potential applications of LLMs in industrial environments for cyberattack detection. While LLMs show promise in detecting cyber threats without prior task-specific training, challenges were noted regarding zero-shot inference on high-dimensional tabular data, which may impact model performance in real-world scenarios.
