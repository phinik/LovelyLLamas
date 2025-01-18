# Weather Report Rewriting: Creative Text Generation

## Overview
This folder includes scripts designed to rewrite weather reports in a more creative and engaging manner.

### Files:
1. **`GPT_local.py`**  
   Utilizes a local AI model (`LLaMA-3.1`) to generate creative and humorous rephrased weather reports. The script employs efficient 4-bit quantization for processing and dynamically adjusts input/output token lengths to enhance text generation.

2. **`UHHGPT_scraper.py`**  
   Automates interactions with the UHH GPT web interface, leveraging Selenium to submit weather reports and retrieve creatively rewritten versions.

3. **`UHHGPT_scraper.py`**
    Uses the OpenAI paid API to generate creative reports with the gpt-4o-mini model

### Approach:
- **Local AI Generation:**  
  A transformer-based model rewrites input text into humorous weather reports with tailored prompts.
  
- **Web-Based Rewriting:**  
  Automates the submission and retrieval of rewritten weather reports from an online GPT-based interface.

- **API-Based Rewriting:**
  Automates the submission and retrieval of JSONL weather reports from the GPT-API interface.
---
