import gradio as gr
from predictions import prediction

demo = gr.Interface(
    fn=prediction,
    inputs= gr.Audio(label="Audio file", type="filepath"),
    outputs= "text",
)

demo.launch(share=True)