from datasets import load_dataset

# Download the dataset
dataset = load_dataset("conll2003",trust_remote_code=True)

# Save locally in a directory
dataset.save_to_disk("../conll2003_local")
print("Dataset downloaded and saved locally at './conll2003_local'")