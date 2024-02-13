import torch
from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids


from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

#----------------------------------------------------------------------------------------------------------------------
# Read context
import pandas as pd

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
sample_size = 10

data_attack = pd.read_csv('/media/work/icarovasconcelos/LLM_Attacks_Recognition/SWaT/SWaT_Dataset_Attack_v0.csv')

# Split the dataframe into two based on the 'Normal/Attack' column
normal_data = data_attack[data_attack['Normal/Attack'] == 'Normal']
attack_data = data_attack[data_attack['Normal/Attack'] == 'Attack']

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
tuples = []
for row in sample.itertuples(index=False):
    data_tuple = tuple(row)
    attributes = [f"{column}: {value}" for column, value in zip(sample.columns, data_tuple)]
    data_string = "The following text contains the physical properties related to a water plant and the water treatment process, as well as network traffic in the testbed. The data of both physical properties and network traffic contains attacks in Cyber Physical Systems: " + " | ".join(attributes) + "\nThe data identifies the text with 'Attack' or 'Normal'?"
    tuples.append(data_string)

#---------------------------------------------------------------------------------------------------------------------

res = generate_text(tuples)
print(res[0]["generated_text"])
'''
for r in res:
    print(r["generated_text"])
    print("\n\n\n")'''

'''res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse.")
'''