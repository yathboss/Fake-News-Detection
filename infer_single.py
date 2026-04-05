import argparse
import sys
import os
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformers import CLIPProcessor
    from models.multimodal_classifier import MultimodalFakeNewsClassifier
    from retrieval.rag_retriever import RealRAGRetriever
except ImportError as e:
    print(f"Error importing modules: {e}. Please ensure requirements are installed.")
    sys.exit(1)

def infer_single(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    
    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("Model checkpoint loaded.")
    else:
        print("Warning: Checkpoint not found. Using untrained multimodal fusion head.")
        
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # RAG Integration
    retriever = RealRAGRetriever()
    evidence = retriever.retrieve(args.text, top_k=2)
    augmented_text = args.text
    if evidence:
        augmented_text += " [EVIDENCE] " + " ".join([e["text"] for e in evidence])
        
    if args.image_path and os.path.exists(args.image_path):
        image = Image.open(args.image_path).convert('RGB')
    else:
        image = Image.new('RGB', (224, 224), color='black')
        
    inputs = processor(
        text=[augmented_text], 
        images=[image], 
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
    
    print(f"\n--- Inference Result ---")
    print(f"Claim: {args.text}")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {prob[pred]:.4f}")
    if evidence:
        print("\nRetrieved Evidence used for RAG Context:")
        for idx, ev in enumerate(evidence):
            print(f"[{idx+1}] {ev['text']} (Score: {ev['score']:.4f})")
    else:
        print("\nNo RAG evidence found or RAG index not built.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Multimodal Inference")
    parser.add_argument("--text", type=str, required=True, help="Text claim to verify")
    parser.add_argument("--image_path", type=str, required=False, default="", help="Path to associated image")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_best.pt")
    
    args = parser.parse_args()
    infer_single(args)
