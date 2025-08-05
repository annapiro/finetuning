import os

from accelerate import Accelerator
from accelerate import load_checkpoint_and_dispatch
from dotenv import load_dotenv
import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, pipeline


load_dotenv()
model_name = os.getenv('MODEL_NAME')


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit inference
    bnb_4bit_compute_dtype="bfloat16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    #torch_dtype=torch.bfloat16,
    device_map="auto",
    #attn_implementation="flash_attention_2",
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
)

#acc = Accelerator()

#model = acc.prepare(model)
# Load text-generation pipeline with automatic sharding
generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
)

while True:
    prompt = input("\nPrompt: ").replace('\\n', '\n')
    out = generator(prompt)
    print(out[0]['generated_text'])
    #inputs = tokenizer.encode(prompt, return_tensors="pt")  # Get tensors
    #input_ids = inputs
    #attention_mask = torch.ones_like(input_ids, dtype=torch.long)


    #for _ in range(300):  # Generate up to 50 tokens
    #output = model.generate(input_ids,
    #                            max_new_tokens=50,
    #                            do_sample=True,
    #                            attention_mask = attention_mask,
    #                            pad_token_id=tokenizer.eos_token_id)
        #new_token = output[:, -1].unsqueeze(0)
        #input_ids = torch.cat([input_ids, new_token], dim=-1)  # Append token
        #new_mask = torch.ones((1, 1), dtype=torch.long).to("cuda")  # Shape: [1, 1]
        #attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        # Decode and print the new token (streaming effect)

    #print(output)
    #print(tokenizer.decode(output), end="", flush=True)
