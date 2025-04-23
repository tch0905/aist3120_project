from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("../../roberta-large-local", add_prefix_space=True )
tokenizer_front = RobertaTokenizerFast.from_pretrained("../../roberta-large-local")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # Important change
        padding="max_length",
        max_length=128,
    )

    # Handle cases where first token starts with space
    for i, sentence in enumerate(examples["tokens"]):
        if len(sentence) > 0 and not sentence[0].startswith(' '):
            # Tokenize the first word separately
            first_word_encoding = tokenizer_front(sentence[0], add_special_tokens=False)
            if len(first_word_encoding['input_ids']) > 0:
                # Replace the first token in the input_ids
                tokenized_inputs['input_ids'][i][1] = first_word_encoding['input_ids'][0]  # Skip [CLS] token
                # Similarly update attention_mask if needed
                tokenized_inputs['attention_mask'][i][1] = 1

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # New word
            else:
                label_ids.append(-100)  # Subword (optional: use label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs