from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import ELMoEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from pathlib import Path
import os
from collections import Counter

# 1. Prepare the dataset (if coming from Hugging Face)
def prepare_conll2003_9class_from_hf():
    from datasets import load_dataset, load_from_disk
    dataset = load_from_disk("../../../conll2003_local")
    
    # Create directory structure
    os.makedirs("data/conll2003_9class", exist_ok=True)
    
    # Mapping from numerical tags to BIO tags
    tag_mapping = {
        0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 
        4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
    }
    
    def save_split(split_data, filename):
        with open(filename, 'w') as f:
            for example in split_data:
                tokens = example['tokens']
                ner_tags = [tag_mapping[tag] for tag in example['ner_tags']]
                pos_tags = example['pos_tags']
                chunk_tags = example['chunk_tags']
                
                for token, pos, chunk, ner in zip(tokens, pos_tags, chunk_tags, ner_tags):
                    f.write(f"{token}\t{pos}\t{chunk}\t{ner}\n")
                f.write("\n")
    
    save_split(dataset['train'], "data/conll2003_9class/train.txt")
    save_split(dataset['validation'], "data/conll2003_9class/dev.txt")
    save_split(dataset['test'], "data/conll2003_9class/test.txt")

# Run this if you need to convert from HF format
prepare_conll2003_9class_from_hf()

# 2. Load the 9-class corpus
columns = {0: 'text', 1: 'pos', 2: 'chunk', 3: 'ner'}
data_folder = Path('./data/conll2003_9class')

corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file='train.txt',
    dev_file='dev.txt',
    test_file='test.txt'
)

# 3. Verify the 9 classes are present
print("Available NER tags in the corpus:")
tag_dictionary = corpus.make_tag_dictionary(tag_type="ner")
print(tag_dictionary.get_items())

# 4. Show label distribution
label_counter = Counter()
for sentence in corpus.train:
    for token in sentence:
        label = token.get_head("ner").value
        label_counter[label] += 1

print("\nLabel distribution in training set:")
for label, count in label_counter.most_common():
    print(f"{label}: {count}")

# 5. Initialize embeddings (ELMo + Flair for best performance)
embeddings = StackedEmbeddings([
    ELMoEmbeddings('original'),
    FlairEmbeddings('news-forward-fast'),
    FlairEmbeddings('news-backward-fast')
])

# 6. Create tagger for 9-class problem
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type="ner",
    use_crf=True
)

# 7. Train the model
trainer = ModelTrainer(tagger, corpus)
trainer.train(
    'resources/taggers/conll2003_9class_elmo',
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=2,
    monitor_test=True,
    patience=5
)

# 8. Evaluation
print("\nFinal Evaluation on Test Set:")
results = tagger.evaluate(corpus.test, gold_label_type="ner")

# Print overall metrics
print(f"\nMicro-averaged F1: {results.main_score:.4f}")
print(f"Precision: {results.precision:.4f}")
print(f"Recall: {results.recall:.4f}")

# Print detailed per-class metrics
print("\nPer-class performance:")
for label in ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'O']:
    if label in results.class_to_f1:
        print(f"{label}:")
        print(f"  F1: {results.class_to_f1[label]:.4f}")
        print(f"  Precision: {results.class_to_precision[label]:.4f}")
        print(f"  Recall: {results.class_to_recall[label]:.4f}")

# 9. Example prediction
print("\nExample prediction:")
sentence = Sentence("Apple Inc. is opening a new store in Berlin, Germany on January 9")
tagger.predict(sentence)
print(sentence.to_tagged_string())