## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:Developing a user-friendly web application that allows users to input text prompts and generate corresponding images using a pre-trained Stable Diffusion model. The application should provide a simple interface for entering prompts and displaying the generated images in real-time, with the option to share the application publicly.

### DESIGN STEPS:

#### STEP 1:
Install: pip install diffusers transformers gradio torch torchvision accelerate safetensors
#### STEP 2:
Import: import gradio as gr, from diffusers import StableDiffusionPipeline, import torch
#### STEP 3:
Load Model: pipe = StableDiffusionPipeline.from_pretrained
#### STEP 4:
Move to GPU: pipe.to("cuda") (if available)
#### STEP 5:
Define Generate Function: def generate_image(prompt): ... return image
#### STEP 6:
Launch Gradio App: gr.Interface(fn=generate_image, inputs=..., outputs=...).launch()

### PROGRAM:
```
!pip install diffusers transformers gradio torch torchvision accelerate safetensors

import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your image prompt", placeholder="e.g. A dragon flying in the sky"),
    outputs=gr.Image(label="Generated Image"),
    title="AI Image Generator",
    description="Demo image generator without any delay. Perfect for prototype submission. No token required.",
    allow_flagging="never"
).launch()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/33d69c00-1437-4a06-be66-340794c98ae4)

### RESULT:
Therefore the code is excuted successfully. 
