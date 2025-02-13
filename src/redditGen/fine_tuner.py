import torch
import pandas as pd
import callbacks
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from langchain import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda

DATASET_PATH = ""
MODEL_NAME = "teknium/OpenHermes-2.5-Mistral-7B"

def tokenize_function(examples):
    formatted_texts = [
        f"Subreddit: {sub}\nTopic: {top}\nTitle: {tit}\nPost: {bod}"
        for sub, top, tit, bod in zip(
            examples['subreddit'], examples['topic'], examples['title'], examples['body']
        )
    ]

    tokenized = tokenizer(
        formatted_texts, padding="max_length", truncation=True, max_length=1024
    )

    tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for training
    return tokenized

def fine_tune():
    df = pd.read_csv(DATASET_PATH)
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        bnb_8bit_compute_dtype=torch.float16,  # Compute in fp16 for efficiency
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically assigns model to GPU/CPU
        quantization_config=bnb_config,
        # torch_dtype=torch.bfloat16,  # TPU needs bfloat16
    )

    dataset = Dataset.from_pandas(df)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Or add a new pad token

    # Tokenize dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Apply LoRA (Efficient Fine-Tuning)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./openhermes-reddit-chain",
        bf16=True,  # Prefer bf16 if supported, otherwise use fp16=True
        optim="adamw_8bit",  # Optimized AdamW for 8-bit training
        learning_rate=1e-5,  # Reduce LR slightly for better stability
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        # evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        callbacks=[
            callbacks.EarlyStoppingOnTrainLoss(patience=10),
            callbacks.SaveBestTrainingLossCallback(),
        ]
    )

    trainer.train()
