
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tools import TOOLS

class ToolRouter:
    def __init__(self, threshold=0.45):
        self.threshold = threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.tool_texts = [t["description"] for t in TOOLS]
        self.tool_embeddings = self.model.encode(
            self.tool_texts,
            normalize_embeddings=True
        )

    def route(self, query: str):
        query_emb = self.model.encode([query], normalize_embeddings=True)

        sims = cosine_similarity(query_emb, self.tool_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        # Debug: print all scores
        print(f"\n[Router Debug]")
        for i, (tool, score) in enumerate(zip(TOOLS, sims)):
            print(f"  {tool['name']}: {score:.3f}")
        print(f"  Threshold: {self.threshold}")
        print(f"  Best: {TOOLS[best_idx]['name']} ({best_score:.3f})\n")

        if best_score < self.threshold:
            return None

        return {
            "tool": TOOLS[best_idx]["name"],
            "confidence": best_score
        }