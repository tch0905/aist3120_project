from transformers import AutoTokenizer, BertForTokenClassification, AutoModelForMaskedLM, RobertaTokenizer, RobertaModel
from transformers.utils import is_offline_mode
from transformers.modeling_utils import init_empty_weights 

# # Download and save the tokenizer locally
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)

# # Save the tokenizer for offline usage
# tokenizer.save_pretrained("../bert-base-cased-local")
# model.save_pretrained("../bert-base-cased-local")
# print("Tokenizer downloaded and saved locally at '../bert-base-cased-local'")

# # Load the tokenizer from the local directory
# tokenizer_local = AutoTokenizer.from_pretrained("../bert-base-cased-local")

# Define the model name
model_name = "FacebookAI/roberta-base"

# Define the local directory where you want to save the model
save_directory = "../roberta-base-local"

tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# Save the tokenizer and model to the local directory
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tokenizer_local = AutoTokenizer.from_pretrained(save_directory)

# Verify by tokenizing a sample text
print(tokenizer_local.tokenize("Hello, how are you?"))