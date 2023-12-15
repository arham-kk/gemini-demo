# Gemini Playground 💬

## Overview

Gemini Playground is an interactive Python application that leverages Google Generative AI and Gradio to provide a platform for experimenting with Gemini Pro and Gemini Pro Vision models. Users can engage in dynamic conversations with the chatbot, utilizing text and image prompts to generate creative and context-aware responses.

## Prerequisites

Before running the code, ensure you have the necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

Make sure to obtain your own Google API key by visiting [GOOGLE API KEY](https://makersuite.google.com/app/apikey) and follow the instructions.

## Usage

1. Set up your Google API key by entering it in the `GOOGLE API KEY` textbox.
2. Optionally provide an image prompt to explore the capabilities of Gemini Pro Vision.
3. Input your text prompt in the `Ask me anything and press Enter` textbox.
4. Adjust various parameters such as temperature, token limit, stop sequences, top-K, and top-P to tailor the model's behavior.
5. Click the "Run" button to initiate the conversation.

## Features

- **Gemini Pro and Gemini Pro Vision**: Choose between text-only (Gemini Pro) or text and image prompts (Gemini Pro Vision).
- **Interactive Chat Interface**: Engage in dynamic conversations with the chatbot using Gradio's intuitive chat interface.
- **Flexible Parameter Adjustment**: Fine-tune model behavior with adjustable parameters like temperature, token limit, and more.

## Parameters

- **GOOGLE API KEY**: Your personal Google API key for accessing the Generative AI models.
- **Image**: Optional image prompt for Gemini Pro Vision.
- **Temperature**: Controls the randomness of the generated content. Higher values lead to more creative responses.
- **Token Limit**: Limit the length of the generated content in tokens.
- **Stop Sequences**: Specify stop sequences to control the generation process.
- **Top-K**: Control the diversity of the output by selecting from the top-K most likely tokens.
- **Top-P**: Control the diversity of the output by selecting tokens with cumulative probabilities up to the top-P threshold.

## Important Note

Ensure you a valid Google API key to use the Gemini Pro and Gemini Pro Vision models.
