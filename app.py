"""
app.py

Streamlit demo app for the animal classifier.

Loads the trained encoder + classifier checkpoints, lets the user
upload an image, and shows the predicted class with confidence scores.

Run with:
    streamlit run app.py
"""

import json

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

import config
from models.embedding_net import EmbeddingNet
from models.classifier import ClassifierHead
from dataset import build_eval_transform


# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once, not on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    device = torch.device(config.DEVICE)

    with open(config.CLASS_NAMES_PATH) as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    encoder = EmbeddingNet(
        backbone_name=config.BACKBONE_NAME,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=False,  # we're loading our own trained weights, not ImageNet ones
        base_channels=config.BASE_CHANNELS,
    ).to(device)
    encoder.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/encoder.pt", map_location=device))
    encoder.eval()

    classifier = ClassifierHead(
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=len(class_to_idx),
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        dropout=config.CLASSIFIER_DROPOUT,
    ).to(device)
    classifier.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/classifier.pt", map_location=device))
    classifier.eval()

    transform = build_eval_transform(config.IMAGE_SIZE)

    return encoder, classifier, transform, idx_to_class, device


@torch.no_grad()
def predict(image, encoder, classifier, transform, idx_to_class, device, top_k=5):
    """Runs a single PIL image through encoder -> classifier, returns top-k predictions."""
    image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    embedding = encoder(image_tensor)
    logits = classifier(embedding)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    top_probs, top_indices = probs.topk(top_k)
    results = [
        (idx_to_class[idx.item()], prob.item())
        for prob, idx in zip(top_probs, top_indices)
    ]
    return results


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Animal Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Animal Species Classifier")
st.write(
    "Upload a photo and the model will predict which of 45 animal species it is. "
    "Built with a Siamese network encoder + classifier head, trained on a "
    "ResNet18 backbone."
)

encoder, classifier, transform, idx_to_class, device = load_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Classifying..."):
            results = predict(image, encoder, classifier, transform, idx_to_class, device, top_k=5)

        st.subheader("Predictions")
        top_class, top_prob = results[0]
        st.success(f"**{top_class.replace('_', ' ').title()}** ({top_prob:.1%} confidence)")

        st.write("Top 5:")
        for class_name, prob in results:
            display_name = class_name.replace("_", " ").title()
            st.write(f"{display_name}")
            st.progress(prob)
else:
    st.info("Upload an image to get a prediction.")

st.divider()
st.caption(
    f"Backbone: {config.BACKBONE_NAME} | Embedding dim: {config.EMBEDDING_DIM} | "
    f"Classes: {len(idx_to_class)}"
)