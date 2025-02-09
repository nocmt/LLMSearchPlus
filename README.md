# LLM Search Plus

Enable LLM to support online search capabilities.

[中文](./README_CN.md) | [English](./README.md)

Currently supported search engines:

- Google
- SearxNG

Local model service:

- LM Studio

Or other services compatible with OpenAI API.

### Usage:

Prerequisites: Python environment is already installed locally, or you can install it from: https://www.anaconda.com/download/success (Miniconda is recommended)

**Clone the project locally:**

    git clone git@github.com:nocmt/LLMSearchPlus.git

**Install dependencies:**

    pip install -r requirements.txt

**Copy or rename .env.template to .env, and modify the configurations within.**

**Run:**

    python main.py

For use in Chat clients, set the URL to: http://127.0.0.1:8100 (you may need to add /v1). The setup is similar to using LM Studio, please test to determine the exact configuration needed.

### Online Search Logic:

Include #search or /search in your message to force enable online search, otherwise the model will determine whether internet search is needed.