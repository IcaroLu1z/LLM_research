from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import pandas as pd
from optimum.bettertransformer import BetterTransformer

#model_id = "mistralai/Mistral-7B-v0.1"
#model_id = "meta-llama/Llama-2-7b-hf"
model_id = "tiiuae/falcon-7b"
#model_id = "01-ai/Yi-6B"

if model_id == "tiiuae/falcon-7b":
    model_pres = torch.bfloat16
elif model_id == "meta-llama/Llama-2-7b-hf":
    model_pres = torch.bfloat16
elif model_id == "mistralai/Mistral-7B-v0.1":
    model_pres = torch.bfloat16
elif model_id == "01-ai/Yi-6B":
    model_pres = torch.bfloat16
    


# Check if a GPU is available
if torch.cuda.is_available():
    # Specify the GPU device
    device = torch.device("cuda:0") 
else:
    device = torch.device("cpu")

print(device)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)
model_ = model.to(dtype=model_pres)
if model_id != "mistralai/Mistral-7B-v0.1" and model_id != "01-ai/Yi-6B":
    model_ = BetterTransformer.transform(model, keep_original_model=False)
model_.to(device)

print("Max Position Embeddings:", config.max_position_embeddings)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_,
    tokenizer=tokenizer,
    torch_dtype=model_pres,
    device=device,
    #trust_remote_code=True,
    #device_map="auto",
)


sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
