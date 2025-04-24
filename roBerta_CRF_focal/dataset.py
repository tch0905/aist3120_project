from datasets import load_from_disk

dataset = load_from_disk("../conll2003_local")
label_names = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_names)