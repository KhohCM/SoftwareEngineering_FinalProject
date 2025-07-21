import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Load tokenizer and model with CUDA support
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {str(e)}")
    raise

# 2. Load and preprocess the dataset
try:
    logger.info("Loading dataset...")
    if not os.path.exists("gordon_ramsay_conversations.txt"):
        raise FileNotFoundError("Dataset file 'gordon_ramsay_conversations.txt' not found!")
    
    with open("gordon_ramsay_conversations.txt", "r", encoding="utf-8") as f:
        conversations = f.read().split("\n\n")

    texts = [conv for conv in conversations if conv.strip()]
    logger.info(f"Loaded {len(texts)} conversations")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

# Create a dataset
dataset = Dataset.from_dict({"text": texts})

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

try:
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
except Exception as e:
    logger.error(f"Error tokenizing dataset: {str(e)}")
    raise

# 3. Split into train and validation
train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

# 4. Define training arguments with CUDA optimization
training_args = TrainingArguments(
    output_dir="gpt2-ramsay-training",  # Changed to a local directory
    overwrite_output_dir=False,  # Allow resuming
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=5000,  # Increased to reduce frequency of saves
    save_total_limit=2,  # Keep only the last 2 checkpoints
    warmup_steps=100,
    logging_dir="gpt2-ramsay-training/logs",  # Changed logging directory
    logging_steps=100,
    learning_rate=5e-5,
    evaluation_strategy="steps",
    fp16=True,
    resume_from_checkpoint=True,  # Resume from the last checkpoint
    load_best_model_at_end=True,  # Load the best model based on eval loss
    metric_for_best_model="eval_loss",  # Use eval loss to determine the best model
    greater_is_better=False,  # Lower eval loss is better
)

# 5. Data collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Initialize Trainer
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
except Exception as e:
    logger.error(f"Error initializing trainer: {str(e)}")
    raise

# 7. Train the model on CUDA
try:
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint="C:/Users/thars/OneDrive/Desktop/gordon-ramsay-chatbot/gpt2-ramsay-training/checkpoint-25000")  # Specify the exact checkpoint path
except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    raise

# 8. Save the fine-tuned model
try:
    model.save_pretrained("gpt2-ramsay-finetuned2")
    tokenizer.save_pretrained("gpt2-ramsay-finetuned2")
    logger.info("Fine-tuning complete, you culinary genius! Model saved to 'gpt2-ramsay-finetuned2'")
except Exception as e:
    logger.error(f"Error saving model: {str(e)}")
    raise