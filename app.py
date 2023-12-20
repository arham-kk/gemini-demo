import os
import time
import uuid
from typing import List, Tuple, Optional, Dict, Union

import google.generativeai as genai
import gradio as gr
from PIL import Image

print("google-generativeai:", genai.__version__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

TITLE = """<h1 align="center">Gemini Playground 💬</h1>"""
SUBTITLE = """<h2 align="center">Play with Gemini Pro and Gemini Pro Vision</h2>"""
DES = """
<div style="text-align: center; display: flex; justify-content: center; align-items: center;">
    <span>Run with your 
        <a href="https://makersuite.google.com/app/apikey">GOOGLE API KEY</a>.
    </span>
</div>
"""

IMAGE_CACHE_DIRECTORY = "/tmp"
IMAGE_WIDTH = 512
CHAT_HISTORY = List[Tuple[Optional[Union[Tuple[str], str]], Optional[str]]]

def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
    return [sequence.strip() for sequence in stop_sequences.split(",")] if stop_sequences else None

def preprocess_image(image: Image.Image) -> Optional[Image.Image]:
    if image:
        image_height = int(image.height * IMAGE_WIDTH / image.width)
        return image.resize((IMAGE_WIDTH, image_height))

def cache_pil_image(image: Image.Image) -> str:
    image_filename = f"{uuid.uuid4()}.jpeg"
    os.makedirs(IMAGE_CACHE_DIRECTORY, exist_ok=True)
    image_path = os.path.join(IMAGE_CACHE_DIRECTORY, image_filename)
    image.save(image_path, "JPEG")
    return image_path

def upload(files: Optional[List[str]], chatbot: CHAT_HISTORY) -> CHAT_HISTORY:
    for file in files:
        image = Image.open(file).convert('RGB')
        image_preview = preprocess_image(image)
        if image_preview:
            # Display a preview of the uploaded image
            gr.Image(image_preview).render()
        image_path = cache_pil_image(image)
        chatbot.append(((image_path,), None))
    return chatbot

def user(text_prompt: str, chatbot: CHAT_HISTORY):
    if text_prompt:
        chatbot.append((text_prompt, None))
    return "", chatbot

def bot(
    google_key: str,
    files: Optional[List[str]],
    temperature: float,
    max_output_tokens: int,
    stop_sequences: str,
    top_k: int,
    top_p: float,
    chatbot: CHAT_HISTORY
):
    if not google_key and not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=google_key if google_key else GOOGLE_API_KEY)
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        stop_sequences=preprocess_stop_sequences(stop_sequences=stop_sequences),
        top_k=top_k,
        top_p=top_p
    )

    text_prompt = [chatbot[-1][0]] if chatbot and chatbot[-1][0] and isinstance(chatbot[-1][0], str) else []
    image_prompt = [preprocess_image(Image.open(file).convert('RGB')) for file in files] if files else []
    model_name = 'gemini-pro-vision' if files else 'gemini-pro'
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(text_prompt + image_prompt, stream=True, generation_config=generation_config)

    chatbot[-1][1] = ""
    for chunk in response:
        for i in range(0, len(chunk.text), 10):
            section = chunk.text[i:i + 10]
            chatbot[-1][1] += section
            time.sleep(0.01)
            yield chatbot

google_key_component = gr.Textbox(
    label="GOOGLE API KEY",
    value="",
    type="password",
    placeholder="...",
    info="Please provide your own GOOGLE_API_KEY for this app",
    visible=GOOGLE_API_KEY is None
)
chatbot_component = gr.Chatbot(
    label='Gemini',
    bubble_full_width=False,
    scale=2,
    height=600
)
text_prompt_component = gr.Textbox(
    placeholder="Message...", show_label=False, autofocus=True, scale=8
)
upload_button_component = gr.UploadButton(
    label="Upload Images", file_count="multiple", file_types=["image"], scale=1
)
run_button_component = gr.Button(value="Run", variant="primary", scale=1)
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

user_inputs = [
    text_prompt_component,
    chatbot_component
]

bot_inputs = [
    google_key_component,
    upload_button_component,
    temperature_component,
    max_output_tokens_component,
    stop_sequences_component,
    top_k_component,
    top_p_component,
    chatbot_component
]

with gr.Blocks() as demo:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    gr.HTML(DES)
    with gr.Column():
        google_key_component.render()
        chatbot_component.render()
        with gr.Row():
            text_prompt_component.render()
            upload_button_component.render()
            run_button_component.render()
        with gr.Accordion("Parameters", open=False):
            temperature_component.render()
            max_output_tokens_component.render()
            stop_sequences_component.render()
            with gr.Accordion("Advanced", open=False):
                top_k_component.render()
                top_p_component.render()

    run_button_component.click(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False
    ).then(
        fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
    )

    text_prompt_component.submit(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False
    ).then(
        fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
    )

    upload_button_component.upload(
        fn=upload,
        inputs=[upload_button_component, chatbot_component],
        outputs=[chatbot_component],
        queue=False
    )

demo.queue(max_size=99).launch(debug=False, show_error=True)
