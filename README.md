# Aist 3120 Final project

## ConLL2003 Dataset

This project utilizes the ConLL2003 dataset for training models in Natural Language Processing tasks, specifically focusing on Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. The dataset is accessible via Hugging Face.

- **Dataset Source**: [ConLL2003 Dataset](https://huggingface.co/datasets/tomaarsen/conll2003)

## Dataset Details

- **Total Rows**: 20,700
- **Split**:
  - **Training**: 14,000 rows
  - **Validation**: 3,250 rows
  - **Test**: 3,450 rows
- **Languages**: English
- **Tasks**: 
  - Token Classification
  - Named Entity Recognition
  - Part-of-Speech Tagging

## Data Format

The dataset includes the following fields for each entry:

- `id`: Unique identifier
- `document_id`: Identifier for the document
- `sentence_id`: Identifier for the sentence
- `tokens`: List of words/tokens
- `pos_tags`: Part-of-speech tags for each token
- `chunk_tags`: Chunking tags for each token
- `ner_tags`: Named entity recognition tags for each token

### Example Entry

```json
{
  "id": 1,
  "document_id": 1,
  "sentence_id": 1,
  "tokens": ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
  "pos_tags": [22, 42, 16, 21, 35, 37, 16, 21, 7],
  "chunk_tags": [11, 21, 11, 12, 21, 22, 11, 12, 0],
  "ner_tags": [3, 0, 7, 0, 0, 0, 7, 0, 0]
}