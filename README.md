# LLM-Based-RBMT

A hybrid English-to-Yoruba translation system combining Rule-Based Machine Translation (RBMT) and a Large Language Model (LLM). This project restructures English sentences according to Yoruba's Subject-Verb-Object (SVO) grammar rules and uses a bilingual dictionary for word translations, aiming for accurate and contextually appropriate translations.

## Features

- **Rule-Based Translation:** Applies Yoruba phrase structure rules to reorder sentences.
- **LLM Integration:** Uses an LLM for restructuring complex grammatical patterns.
- **Bilingual Dictionary:** Translates words using a dictionary-based approach.
- **Evaluation Metrics:** Assesses translation quality with BLEU and SacreBLEU scores.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cyberguru1/LLM-Based-RBMT.git
   cd LLM-Based-RBMT
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Translate an English sentence:

```bash
python en-yo.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

An LLM based rule based machine translation utilizing gemini
