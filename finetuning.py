import atexit
import os

from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, pipeline, Trainer, TrainingArguments

load_dotenv()
model_name = os.getenv('MODEL_NAME')
dataset_path = os.getenv('DATASET_PATH')
output_path = os.getenv('OUTPUT_PATH')


def cleanup():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def preprocess(batch):
    tokenized_batch = tokenizer(
        batch["code"],
        truncation=True,
        padding="max_length",
        max_length = 512,  # TODO: too short for code snippets
        return_tensors="pt",
    )
    tokenized_batch["labels"] = tokenized_batch["input_ids"].clone()
    return tokenized_batch


if __name__ == "__main__":
    # register the cleanup function that will run at exit
    atexit.register(cleanup)

    # check device status
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"}')


    # --- INITIALIZATION ---
    # Quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        # device_map="auto",
        torch_dtype=torch.float16,
    )
    # model.print_trainable_parameters()  # TODO: find some way to check the nr of params

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # PEFT (LoRA) TODO: revise hyperparameters
    lora_config = LoraConfig(
        r=16,
        lora_alpha = 32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model with all the configurations
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    # need to set gradient checkpointing to silence a warning
    model.gradient_checkpointing_disable()
    # leads to a reduction in memory with a slight decrease in speed
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()


    # --- DATASET ---
    dataset = load_dataset("json", data_files=dataset_path)
    print("\nLoaded dataset:")
    print(dataset)

    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    print("\nSplit dataset:")
    print(dataset)

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        # num_proc = 4,
    )
    print("\nTokenized dataset:")
    print(tokenized_dataset)

    # test collator
    # sample_batch = [tokenized_dataset["train"][i] for i in range(2)]
    # collated = data_collator(sample_batch)
    # print("Collated keys:", collated.keys())

    # some dataset stats
    lengths = [len(x["input_ids"]) for x in tokenized_dataset["train"]]
    print(f"\nLength stats - Max: {max(lengths)}, Min: {min(lengths)}, Avg: {sum(lengths)/len(lengths)}")


    # --- TRAINING ---
    training_args = TrainingArguments(
        output_dir=output_path,
        report_to="none",
        per_device_train_batch_size=2,
        remove_unused_columns=False,
        fp16=True,
        # eval_strategy="epoch",
        # learning_rate=2e-5,
        # weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=tokenized_dataset["train"],
        train_dataset=tokenized_dataset["train"].select(range(100)),
        # eval_dataset=tokenized_dataset["test"],
        eval_dataset=tokenized_dataset["test"].select(range(20)),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
