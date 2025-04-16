import torch

def augment_batch_with_random_concat(batch, tokenizer, max_length=512, pad_label_id=-100):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size = input_ids.size(0)

    # Generate a random permutation of indices for pairing
    permuted_indices = torch.randperm(batch_size)

    # New containers for augmented samples
    new_input_ids = []
    new_attention_masks = []
    new_labels = []

    for i in range(batch_size):
        # Get original and random sample
        original_input = input_ids[i]
        original_mask = attention_mask[i]
        original_label = labels[i]

        random_input = input_ids[permuted_indices[i]][1:]
        random_mask = attention_mask[permuted_indices[i]][1:]
        random_label = labels[permuted_indices[i]][1:]

        # Remove padding from input and label using attention_mask
        original_input_trimmed = original_input[original_mask.bool()]
        random_input_trimmed = random_input[random_mask.bool()]

        original_label_trimmed = original_label[original_mask.bool()]
        random_label_trimmed = random_label[random_mask.bool()]

        # Concatenate input and label
        concat_input = torch.cat([original_input_trimmed, random_input_trimmed], dim=0)
        concat_label = torch.cat([original_label_trimmed, random_label_trimmed], dim=0)

        # Truncate if needed
        concat_input = concat_input[:max_length]
        concat_label = concat_label[:max_length]

        # Create attention mask
        concat_attention_mask = torch.ones_like(concat_input)

        # Pad input, label, and attention mask to max_length
        pad_len = max_length - concat_input.size(0)
        if pad_len > 0:
            concat_input = torch.cat([concat_input, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
            concat_attention_mask = torch.cat([concat_attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            concat_label = torch.cat([concat_label, torch.full((pad_len,), pad_label_id, dtype=torch.long)])

        # Append to new batch
        new_input_ids.append(concat_input)
        new_attention_masks.append(concat_attention_mask)
        new_labels.append(concat_label)

    # Stack into tensors
    augmented_batch = {
        "input_ids": torch.stack(new_input_ids).to(input_ids.device),
        "attention_mask": torch.stack(new_attention_masks).to(attention_mask.device),
        "labels": torch.stack(new_labels).to(labels.device)
    }

    return augmented_batch