import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Reports as DataFrames
falcon_cr = pd.DataFrame({
    'Metrics': ['Precision', 'Recall', 'F1-score', 'Accuracy'],
    'Normal': [0.64, 0.68, 0.66, 0.65],
    'Attack': [0.66, 0.61, 0.63, 0.65]
}).set_index('Metrics')

llama_cr = pd.DataFrame({
    'Metrics': ['Precision', 'Recall', 'F1-score', 'Accuracy'],
    'Normal': [0.90, 0.36, 0.51, 0.66],
    'Attack': [0.60, 0.96, 0.74, 0.66]
}).set_index('Metrics')

# Confusion Matrices as numpy arrays
falcon_cm = np.array([[68, 32], [39, 61]])
llama_cm = np.array([[36, 64], [4, 96]])

# Plotting Falcon metrics
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))

# Plotting the classification report heatmap for Falcon
sns.heatmap(falcon_cr, annot=True, ax=axes1[0], cmap='Blues', cbar=False)
axes1[0].set_title('Classification Report - Falcon')
axes1[0].set_xlabel('Classes')
axes1[0].set_ylabel('Metrics')
axes1[0].set_xticklabels(['Normal', 'Attack'])

# Plotting the confusion matrix for Falcon
sns.heatmap(falcon_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], ax=axes1[1])
axes1[1].set_xlabel('Predicted')
axes1[1].set_ylabel('True')
axes1[1].set_title('Confusion Matrix - Falcon')

# Increase font size in heatmap annotations
for text in axes1[0].texts:
    text.set_size(20)
for text in axes1[1].texts:
    text.set_size(20)

for ax in axes1:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=15)    

# Adjust layout and save the figure
plt.tight_layout()
fig1.savefig('QA_Falcon.png')

# Plotting Llama metrics
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))

# Plotting the classification report heatmap for Llama
sns.heatmap(llama_cr, annot=True, ax=axes2[0], cmap='Blues', cbar=False)
axes2[0].set_title('Classification Report - Llama')
axes2[0].set_xlabel('Classes')
axes2[0].set_ylabel('Metrics')
axes2[0].set_xticklabels(['Normal', 'Attack'])

# Plotting the confusion matrix for Llama
sns.heatmap(llama_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], ax=axes2[1])
axes2[1].set_xlabel('Predicted')
axes2[1].set_ylabel('True')
axes2[1].set_title('Confusion Matrix - Llama')

for text in axes2[0].texts:
    text.set_size(20)
for text in axes2[1].texts:
    text.set_size(20)

# Increase font size
for ax in axes2:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=15)

# Adjust layout and save the figure
plt.tight_layout()
fig2.savefig('QA_Llama.png')

print("Concluido.")
