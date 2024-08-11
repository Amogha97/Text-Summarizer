from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import os
import torch


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = torch.device("cpu")
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_t5)
        
        # Loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Sampling a smaller subset of the dataset
        train_subset = dataset_samsum_pt["train"].select(range(1000))  # First 1000 samples
        eval_subset = dataset_samsum_pt["validation"].select(range(200))  # First 200 samples
        
        
        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
        #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # ) 

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=1, 
            warmup_steps=500,
            per_device_train_batch_size=8,  
            per_device_eval_batch_size=8, 
            weight_decay=0.01, 
            logging_steps=50,
            evaluation_strategy='steps', 
            eval_steps=500, 
            save_steps=500, 
            gradient_accumulation_steps=4, 
            fp16=False
        ) 

        trainer = Trainer(model=model_t5, args=trainer_args,
                          tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=train_subset, 
                          eval_dataset=eval_subset)
        
        trainer.train()

        ## Save model
        model_t5.save_pretrained(os.path.join(self.config.root_dir,"t5-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
