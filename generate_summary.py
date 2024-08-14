import sys
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Adding the project directory to the system path
project_directory = "/Users/amoghagadde/Desktop/Amogha/Projects/Data_Science/Text-Summarizer/src"
sys.path.append(project_directory)

# Load the pre-trained model and tokenizer
def load_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

# Function to generate summary
def generate_summary(model, tokenizer, text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    model_path = "/Users/amoghagadde/Desktop/Amogha/Projects/Data_Science/Text-Summarizer/artifacts/model_trainer/t5-samsum-model" 
    tokenizer_path = "/Users/amoghagadde/Desktop/Amogha/Projects/Data_Science/Text-Summarizer/artifacts/model_trainer/tokenizer" 
    input_text = sys.argv[1]  # Take input text from command line

    # Load the model and tokenizer
    model, tokenizer = load_model(model_path, tokenizer_path)

    # Generate the summary
    summary = generate_summary(model, tokenizer, input_text)
    print("Summary:", summary)
