from transformers import AutoTokenizer

# Download and save the tokenizer locally
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Save the tokenizer for offline usage
tokenizer.save_pretrained("../bert-base-cased-local")
print("Tokenizer downloaded and saved locally at '../bert-base-cased-local'")

# Load the tokenizer from the local directory
tokenizer_local = AutoTokenizer.from_pretrained("../bert-base-cased-local")

# Verify by tokenizing a sample text
print(tokenizer_local.tokenize("Hello, how are you?"))