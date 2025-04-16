import numpy as np
from transformers import AutoTokenizer

def mask_tokens_in_dataset(dataset, tokenizer, mask_prob=0.15, random_token_prob=0.1, unchanged_prob=0.1):
    """
    Augment dataset by randomly masking tokens (similar to BERT pretraining)
    
    Args:
        dataset: The dataset to augment
        tokenizer: Preloaded tokenizer instance
        mask_prob: Probability of masking a token (default: 15%)
        random_token_prob: Probability of replacing with random token when masking (default: 10%)
        unchanged_prob: Probability of keeping original token when masking (default: 10%)
    
    Returns:
        Augmented dataset with masked tokens
    """
    def mask_example(example):
        input_ids = example["input_ids"]
        masked_input_ids = input_ids.copy()
        
        # Determine which tokens to mask (ignore special tokens)
        maskable_positions = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in tokenizer.all_special_ids
        ]
        
        # Randomly select tokens to mask (at least 1)
        num_to_mask = max(1, int(len(maskable_positions) * mask_prob))
        masked_indices = np.random.choice(
            maskable_positions,
            size=num_to_mask,
            replace=False
        )
        
        for idx in masked_indices:
            rand = np.random.random()
            if rand < random_token_prob:
                # Replace with random token
                masked_input_ids[idx] = np.random.randint(0, len(tokenizer))
            elif rand < random_token_prob + unchanged_prob:
                # Keep original token
                pass
            else:
                # Replace with [MASK]
                masked_input_ids[idx] = tokenizer.mask_token_id
        
        return {"input_ids": masked_input_ids}
    
    return dataset.map(mask_example)