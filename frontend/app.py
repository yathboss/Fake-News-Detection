from __future__ import annotations

import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")

st.set_page_config(page_title="Multimodal Fake News Detector", page_icon="📰", layout="wide")

st.title("Multimodal Fake News Detection using RAG")
st.caption("Targeting MMFakeBench | Phase 2 prototype: text + image claim -> evidence retrieval -> multimodal verdict")

default_claim = "Climate change is a hoax invented by scientists."
claim = st.text_area("Enter a text claim", value=default_claim, height=140)
image_file = st.file_uploader("Upload an associated image (optional for now, multimodal ready)", type=["png", "jpg", "jpeg"])
top_k = st.slider("Number of evidence snippets", min_value=1, max_value=5, value=3)

if st.button("Verify Multimodal Claim", type="primary", use_container_width=True):
    if not claim.strip():
        st.warning("Please enter a text claim first.")
    else:
        with st.spinner("Retrieving evidence and asking the multimodal model..."):
            
            files = {}
            if image_file is not None:
                files["image"] = (image_file.name, image_file.getvalue(), image_file.type)
            
            data = {
                "claim": claim.strip(), 
                "top_k": top_k
            }
            
            response = requests.post(
                f"{API_BASE_URL}/verify",
                data=data,
                files=files if files else None,
                timeout=180,
            )

        if response.ok:
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Label", data["predicted_label"])
            col2.metric("Confidence", f"{data['confidence']:.2f}")
            col3.metric("Model Used", data["model_used"])

            st.subheader("Top Evidence Retrieved via FAISS")
            if not data.get("evidence"):
                st.write("No RAG evidence found. Did you build the FAISS index?")
            
            for item in data.get("evidence", []):
                with st.container(border=True):
                    st.markdown(f"**#{item['rank']}** Score: {item['score']:.4f}")
                    st.write(item["text"])

        else:
            st.error(f"Backend error: {response.status_code}")
            try:
                st.json(response.json())
            except Exception:
                st.text(response.text)
