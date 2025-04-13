import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.set_page_config(page_title="Anime Image Generator", layout="centered")

# Load model (cached so it doesn’t reload every time)
@st.cache_resource
def load_model():
    model_id = "andite/anything-v4.0"  # or hakurei/waifu-diffusion
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

st.title("🎨 Anime Image Generator")
prompt = st.text_input("Enter your anime prompt", "1girl, long hair, blue eyes, forest background")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Anime Image", use_column_width=True)
