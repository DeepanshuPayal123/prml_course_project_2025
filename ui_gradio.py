import gradio as gr
from PIL import Image
import os
import torch

from Experiments.Resnet50_classification import retrieve as model1_retrieve
from Model.centroid_app import retrieve as model2_retrieve


def model_1(image, num_images=3):
    retrived_images = model1_retrieve(image, k=int(num_images))
    return retrived_images

def model_2(image, num_images=3):
    retrived_images = model2_retrieve(image, k=int(num_images))
    return retrived_images

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="violet")) as demo:
    gr.Markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🔍 Image Retrieval System</h1>
    <p style='text-align: center; font-size: 18px; color: #34495e;'>Compare two image retrieval models: <b>RiyalNet</b> (High Accuracy) and <b>QuickNet</b> (Fast Runtime)</p>
    """)

    with gr.Tabs():
        with gr.Tab("💡 RiyalNet - High Accuracy (97%)"):
            with gr.Row():
                with gr.Column(scale=1):
                    query_image_1 = gr.Image(label="Upload Query Image")
                    num_images_1 = gr.Number(value=3, label="Number of Images to Retrieve")
                    btn1 = gr.Button("Retrieve Images")
                with gr.Column(scale=2):
                    output_gallery_1 = gr.Gallery(label="Retrieved Images", type="pil")


            btn1.click(fn=model_1, inputs=[query_image_1, num_images_1], outputs=output_gallery_1)

        with gr.Tab("⚡ QuickNet - Fast Runtime"):
            with gr.Row():
                with gr.Column(scale=1):
                    query_image_2 = gr.Image(label="Upload Query Image")
                    num_images_2 = gr.Number(value=3, label="Number of Images to Retrieve")
                    btn2 = gr.Button("Retrieve Images")
                with gr.Column(scale=2):
                    output_gallery_2 = gr.Gallery(label="Retrieved Images", type="pil")

            btn2.click(fn=model_2, inputs=[query_image_2, num_images_2], outputs=output_gallery_2)

if __name__ == "__main__":
    demo.launch()
