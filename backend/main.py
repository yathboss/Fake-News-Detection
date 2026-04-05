import os
import sys
import io
import torch
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.multimodal_classifier import MultimodalFakeNewsClassifier
    from transformers import CLIPProcessor
    from retrieval.rag_retriever import RealRAGRetriever
except ImportError as e:
    print(f"Error importing modules: {e}")

app = FastAPI(
    title="Multimodal Fake News Detection API",
    description="Text + Image claim -> FAISS retrieval -> CLIP Fusion Verdict",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Multimodal model on {device}...")

try:
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    checkpoint_path = "checkpoints/model_best.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model checkpoint loaded.")
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    retriever = RealRAGRetriever()
except Exception as e:
    print(f"Warning: Model initialization failed. {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/verify")
async def verify(
    claim: str = Form(..., description="The text claim to verify"),
    top_k: int = Form(3, description="RAG top-k evidence to fetch"),
    image: UploadFile = File(None, description="Optional image payload")
):
    if len(claim.strip()) < 3:
        raise HTTPException(status_code=400, detail="Claim is too short.")

    try:
        # 1. Retrieve text evidence via FAISS RAG
        evidence = retriever.retrieve(claim, top_k=top_k)
        augmented_text = claim
        if evidence:
            augmented_text += " [EVIDENCE] " + " ".join([e["text"] for e in evidence])

        # 2. Parse Image Bytes
        img_obj = Image.new('RGB', (224, 224), color='black')
        if image and image.filename:
            content = await image.read()
            img_obj = Image.open(io.BytesIO(content)).convert('RGB')

        # 3. Process Multimodal pass
        inputs = processor(
            text=[augmented_text], 
            images=[img_obj], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(device)

        with torch.no_grad():
            logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=inputs.pixel_values)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        pred = prob.argmax()
        label = "Fake" if pred == 1 else "Real"
        
        # 4. Format evidence packet
        evidence_list = []
        for idx, ev in enumerate(evidence):
            evidence_list.append({
                "rank": idx + 1,
                "text": ev["text"],
                "score": float(ev["score"])
            })

        return {
            "predicted_label": label,
            "confidence": float(prob[pred]),
            "evidence": evidence_list,
            "model_used": "CLIP Multimodal Fusion + FAISS RAG"
        }
    except Exception as exc:
        print(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
