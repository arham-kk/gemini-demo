import time
from typing import List, Tuple, Optional
import google.generativeai as genai
import gradio as gr
from PIL import Image

# HTML components for the UI
TITLE = """<h1 align="center">Gemini Playground 💬</h1>"""
SUBTITLE = """<h2 align="center">Play with Gemini Pro and Gemini Pro Vision</h2>"""
DUPLICATE = """
<div style="text-align: center; display: flex; justify-content: center; align-items: center;">
    <span>Run with your own Google Api Key: 
        <a href="https://makersuite.google.com/app/apikey">GOOGLE API KEY</a>.
    </span>
</div>
"""

# Function to preprocess stop sequences
def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
    if not stop_sequences:
        return None
    return [sequence.strip() for sequence in stop_sequences.split(",")]

# Function for user interaction
def user(text_prompt: str, chatbot: List[Tuple[str, str]]):
    return "", chatbot + [[text_prompt, None]]

# Function for bot interaction
def bot(
    google_key: str,
    image_prompt: Optional[Image.Image],
    temperature: float,
    max_output_tokens: int,
    stop_sequences: str,
    top_k: int,
    top_p: float,
    chatbot: List[Tuple[str, str]]
):
    # Check if Google API key is provided
    if not google_key:
        raise ValueError("GOOGLE_API_KEY is not set.")

    # Set up Google Generative AI
    text_prompt = chatbot[-1][0]
    genai.configure(api_key=google_key)
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        stop_sequences=preprocess_stop_sequences(stop_sequences=stop_sequences),
        top_k=top_k,
        top_p=top_p
    )

    # Generate content based on text or image prompt
    if image_prompt is None:
        model = genai.GenerativeModel('gemini-pro')
    else:
        model = genai.GenerativeModel('gemini-pro-vision')

    response = model.generate_content(
        [text_prompt, image_prompt] if image_prompt else text_prompt,
        stream=True,
        generation_config=generation_config
    )
    response.resolve()

    # Streaming effect
    chatbot[-1][1] = ""
    for chunk in response:
        for i in range(0, len(chunk.text), 10):
            section = chunk.text[i:i + 10]
            chatbot[-1][1] += section
            time.sleep(0.01)
            yield chatbot

# Gradio components for the UI
google_key_component = gr.Textbox(
    label="GOOGLE API KEY",
    value="",
    type="password",
    placeholder="...",
    info="Please provide your own GOOGLE API KEY",
)

image_prompt_component = gr.Image(type="pil", label="Image", scale=1)
chatbot_component = gr.Chatbot(
    label='Gemini',
    bubble_full_width=False,
    scale=2
)
text_prompt_component = gr.Textbox(
    placeholder="Hi there!",
    label="Ask me anything and press Enter"
)
run_button_component = gr.Button()
temperature_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.4,
    step=0.05,
    label="Temperature",
)
max_output_tokens_component = gr.Slider(
    minimum=1,
    maximum=2048,
    value=1024,
    step=1,
    label="Token limit",
)
stop_sequences_component = gr.Textbox(
    label="Add stop sequence",
    value="",
    type="text",
    placeholder="STOP, END",
)
top_k_component = gr.Slider(
    minimum=1,
    maximum=40,
    value=32,
    step=1,
    label="Top-K",
)
top_p_component = gr.Slider(
    minimum=0,
    maximum=1,
    value=1,
    step=0.01,
    label="Top-P",
)

# Organize UI components using Gradio Blocks
user_inputs = [
    text_prompt_component,
    chatbot_component
]

bot_inputs = [
    google_key_component,
    image_prompt_component,
    temperature_component,
    max_output_tokens_component,
    stop_sequences_component,
    top_k_component,
    top_p_component,
    chatbot_component
]

# Define Gradio UI with HTML components and interaction logic
with gr.Blocks() as demo:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    gr.HTML(DUPLICATE)
    with gr.Column():
        google_key_component.render()
        with gr.Row():
            image_prompt_component.render()
            chatbot_component.render()
        text_prompt_component.render()
        run_button_component.render()
        with gr.Accordion("Parameters", open=False):
            temperature_component.render()
            max_output_tokens_component.render()
            stop_sequences_component.render()
            with gr.Accordion("Advanced", open=False):
                top_k_component.render()
                top_p_component.render()

    # Define interaction logic for the "Run" button
    run_button_component.click(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False
    ).then(
        fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
    )

    # Define interaction logic for submitting the text prompt
    text_prompt_component.submit(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False
    ).then(
        fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
    )

# Launch the Gradio UI
demo.queue(max_size=99).launch(debug=True)
