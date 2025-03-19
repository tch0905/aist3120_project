from datasets import load_dataset



# Load the dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

for train_set in dataset['train']:

    print(train_set['tokens'])
    print(train_set['pos_tags'])
    print(train_set['chunk_tags'])
    print(train_set['ner_tags'])
    pass
