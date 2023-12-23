from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoConfig
import transformers
import torch
import pandas as pd
from optimum.bettertransformer import BetterTransformer

#---------------------------------------------------------------------------------------------------------------------
# Chose the model
#model_id = "mistralai/Mistral-7B-v0.1"
#model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id = "tiiuae/falcon-7b"
#model_id = "01-ai/Yi-6B"
#model_id = "NousResearch/Yarn-Mistral-7b-128k"

#---------------------------------------------------------------------------------------------------------------------
# Check if a GPU is available
if torch.cuda.is_available():
    # Specify the GPU device
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#---------------------------------------------------------------------------------------------------------------------
# Read context
context_path = "/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SwatDataDescription.txt" 
paper_path = "/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SWaT_Paper.txt" 
context = ""
paper = ""
with open(context_path, "r") as file:
    context = file.read()

with open(paper_path, "r") as file:
    paper = file.read()

#---------------------------------------------------------------------------------------------------------------------
# Read and process data
data_attack = pd.read_csv('/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SWaT_Dataset_Attack_v0.csv')

# Split the dataframe into two based on the 'Normal/Attack' column
normal_data = data_attack[data_attack['Normal/Attack'] == 'Normal']
attack_data = data_attack[data_attack['Normal/Attack'] == 'Attack']

# Sample the same number of instances from each class (let's say 100 from each class)
sample_size = 5

sample_normal = normal_data.sample(sample_size, random_state=42)
sample_attack = attack_data.sample(sample_size, random_state=42)

# Combine the sampled dataframes back into one
sample = pd.concat([sample_normal, sample_attack])

# Shuffle the combined dataframe to mix the samples
sample = sample.sample(frac=1, random_state=42)

# Extract the target labels (y)
y = sample['Normal/Attack']

# Drop the class column to get the input features
sample = sample.drop(labels='Normal/Attack', axis=1)

#---------------------------------------------------------------------------------------------------------------------
# Preparing input sentences
questions = []
for row in sample.itertuples(index=False):
    data_tuple = tuple(row)
    attributes = [f"{column}: {value}" for column, value in zip(sample.columns, data_tuple)]
    data_string = "|".join(attributes)
    questions.append(data_string)
#---------------------------------------------------------------------------------------------------------------------
# Preparing the model
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_id, config=config)
model = BetterTransformer.transform(model, keep_original_model=False)
model.to(dtype=torch.bfloat16)


# Find the maximum sequence length in your input data
max_sequence_length = max(len(tokenizer.encode(question)) for question in questions)

# Set the max_length parameter for the tokenizer
tokenizer_max_length = 512  # Set your desired maximum length here
max_length = min(max_sequence_length, tokenizer_max_length)

print(max_length, max_sequence_length, tokenizer_max_length)
#---------------------------------------------------------------------------------------------------------------------


pipeline = transformers.pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    batch_size=1,
    device=device,
    #trust_remote_code=True,
    #device_map="auto",
)

q = 'What is the name of the water plant?'
a = 'The water plant is called SWaT.'

sequences = pipeline(
    question = q,
    context = a,
    #top_k = 1,
    #max_length = max_length,
)   

print(sequences)

'''for seq, y_predict in zip(sequences, y):
    print(f"Resultado: {seq['score']} || {seq['answer']} || true: {y_predict}")'''
