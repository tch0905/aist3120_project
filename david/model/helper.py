from transformers import AutoTokenizer, AutoModelForMaskedLM

# Define the model name
model_name = "FacebookAI/roberta-base"

# Define the local directory where you want to save the model
save_directory = "../../roberta-base-local"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save the tokenizer and model to the local directory
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")