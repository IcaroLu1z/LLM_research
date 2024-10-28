import seaborn as sns
import os
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from optimum.bettertransformer import BetterTransformer
import time
import docx2txt

#---------------------------------------------------------------------------------------------------------------------
# Chose the model
#model_id = "mistralai/Mistral-7B-v0.1"
model_id = "meta-llama/Llama-2-7b-chat-hf"
#model_id = "tiiuae/falcon-7b"
#model_id = "01-ai/Yi-6B"
#model_id = "NousResearch/Yarn-Mistral-7b-128k"


if model_id == "01-ai/Yi-6B" or model_id == "NousResearch/Yarn-Mistral-7b-128k":
    model_pres = torch.bfloat16
    trust_remote_code = True
else:
    trust_remote_code = False
    model_pres = torch.bfloat16

# Sample the same number of instances from each class
sample_size = 10
#---------------------------------------------------------------------------------------------------------------------
# Check if a GPU is available
if torch.cuda.is_available():
    # Specify the GPU device
    device = torch.device("cuda:0")
else:   
    device = torch.device("cpu")

print(device)
#---------------------------------------------------------------------------------------------------------------------
# Read context
context_path = '/media/work/icarovasconcelos/LLM_Attacks_Recognition/TON_IoT/Description_stats_datasets/Description_stats_Network_dataset/Network Features-Description.docx' 
paper_path = "/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SWaT_Paper.txt" 
context = ""
paper = ""

context = docx2txt.process(context_path)

#---------------------------------------------------------------------------------------------------------------------
# Read and process data
data_attack = pd.read_csv('/media/work/icarovasconcelos/LLM_Attacks_Recognition/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')

data_attack['label'] = data_attack['label'].replace({0: 'Normal', 1: 'Attack'})

# Split the dataframe into two based on the 'Normal/Attack' column
normal_data = data_attack[data_attack['label'] == 'Normal']
attack_data = data_attack[data_attack['label'] == 'Attack']

sample_normal = normal_data.sample(sample_size, random_state=42)
sample_attack = attack_data.sample(sample_size, random_state=42)

# Combine the sampled dataframes back into one
sample = pd.concat([sample_normal, sample_attack])

# Shuffle the combined dataframe to mix the samples
sample = sample.sample(frac=1, random_state=42)

# Extract the target labels (y)
y = sample['label']

# Drop the class column to get the input features
sample = sample.drop(labels='label', axis=1)

#---------------------------------------------------------------------------------------------------------------------
# Preparing input sentences
tuples = []
for row in sample.itertuples(index=False):
    data_tuple = tuple(row)
    attributes = [f"{column}: {value}" for column, value in zip(sample.columns, data_tuple)]
    data_string = " | ".join(attributes) + "\n" + context
    tuples.append(data_string)

#---------------------------------------------------------------------------------------------------------------------
# Preparing model
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=trust_remote_code,
    model_max_length=4096, #Necessario para o llama
    )
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code)
model_ = model.to(dtype=model_pres)
if model_id != "mistralai/Mistral-7B-v0.1" and model_id != "01-ai/Yi-6B":
    model_ = BetterTransformer.transform(model_, keep_original_model=False)
model_.to(device)

#---------------------------------------------------------------------------------------------------------------------
pipeline = transformers.pipeline(
    "zero-shot-classification",
    model=model_,
    tokenizer=tokenizer,
    framework="pt",
    batch_size=1,
    torch_dtype=model_pres,
    device=device,
    #trust_remote_code=True,
    #device_map="auto",
)

start_time = time.time()

sequences = pipeline(
    sequences = tuples,
    candidate_labels = ['Normal', 'Attack'],
)

#---------------------------------------------------------------------------------------------------------------------
end_time = time.time()
execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

#---------------------------------------------------------------------------------------------------------------------
predicted_labels = []
true_labels = []

for seq, y_p in zip(sequences, y):
    formatted_scores = [f"{score:.2f}" for score in seq['scores']]
    predicted_label = seq['labels'][0]  # Taking the top predicted label
    true_labels.append(y_p)
    predicted_labels.append(predicted_label)
    print(f"Resultado: {predicted_label} || Scores: {formatted_scores} || True: {y_p}")

clf_report = classification_report(y_true=true_labels, y_pred=predicted_labels, target_names=['Normal', 'Attack'])
# Using confusion_matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=['Attack', 'Normal'])


pattern = re.compile(r'/(.*?)-7b')
model_family = re.findall(pattern, model_id)[0]
model_name = model_id.split('/')[-1]
path = f'/media/work/icarovasconcelos/LLM_Attacks_Recognition/HugginFaceModels/{model_family}/'
os.makedirs(path, exist_ok=True)

with open(f'/media/work/icarovasconcelos/LLM_Attacks_Recognition/HugginFaceModels/{model_family}/' + f'ZS_TON_IoT_{model_name}_{str(sample_size*2)}_Samples.txt', 'w') as file:
    file.write("\nClassification Report:\n")
    file.write(clf_report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))
    file.write(f"\nExecution time: {int(hours)}:{int(minutes)}:{seconds:.2f} (h:mm:ss)")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

clf_report_df = pd.DataFrame(classification_report(true_labels, predicted_labels, target_names=['Normal', 'Attack'], output_dict=True)).iloc[:-1, :].T

# Plot the heatmap for the classification report
sns.heatmap(clf_report_df, annot=True, ax=axes[0], cmap='Blues')
axes[0].set_title('Classification Report')


# Plotting the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Attack', 'Normal'], yticklabels=['Attack', 'Normal'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix')

# Adjust layout
plt.tight_layout()

# Save the figure
figName = f'ZSComparison_TON_IoT_{model_name}__{str(sample_size*2)}_Samples.png'
plt.savefig(f'/media/work/icarovasconcelos/LLM_Attacks_Recognition/HugginFaceModels/{model_family}/' + figName)

print("Concluido.")