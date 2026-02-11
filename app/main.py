from __future__ import annotations

from typing import List, Union, Optional
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embed-service")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

app = FastAPI(title="Embedding Service", version="1.0.0")

model: Optional[SentenceTransformer] = None


class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]] = Field(
        ...,
        description="A single string or a list of strings to embed."
    )
    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize embeddings (often recommended for similarity)."
    )

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("inputs must be a non-empty string.")
        else:
            if len(v) == 0:
                raise ValueError("inputs must be a non-empty list of strings.")
            for i, s in enumerate(v):
                if not isinstance(s, str) or not s.strip():
                    raise ValueError(f"inputs[{i}] must be a non-empty string.")
        return v


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Embed Service</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; }
    textarea { width: 100%; height: 140px; font-size: 14px; }
    button { padding: 10px 14px; font-size: 14px; cursor: pointer; }
    pre { background: #111; color: #eee; padding: 14px; border-radius: 10px; overflow:auto; }
    .row { display:flex; gap: 10px; align-items:center; flex-wrap: wrap; }
    .muted { color:#666; font-size: 13px; }
    .links a { text-decoration: none; padding: 6px 10px; border: 1px solid #ddd; border-radius: 999px; color:#111; }
    .links a:hover { background: #f3f3f3; }
    .topbar { display:flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  </style>
</head>
<body>

  <div class="topbar">
    <div>
      <h1 style="margin:0;">Embedding UI</h1>
      <p class="muted" style="margin-top:6px;">One line = one input. Click “Embed”.</p>
    </div>

    <div class="links row" aria-label="Quick links">
      <a href="/docs" target="_blank" rel="noopener">Swagger (/docs)</a>
      <a href="/redoc" target="_blank" rel="noopener">ReDoc (/redoc)</a>
      <a href="/health" target="_blank" rel="noopener">Health (/health)</a>
      <a href="/openapi.json" target="_blank" rel="noopener">OpenAPI JSON</a>
    </div>
  </div>

  <textarea id="txt" placeholder="hello world&#10;fastapi embeddings"></textarea>

  <div class="row" style="margin: 10px 0;">
    <label><input type="checkbox" id="norm" checked> normalize</label>
    <label>show first <input type="number" id="n" value="12" min="1" max="100" style="width:70px;"> values</label>
    <button onclick="embed()">Embed</button>
  </div>

  <div id="meta" class="muted"></div>
  <pre id="out">Waiting...</pre>

<script>
async function embed() {
  const lines = document.getElementById('txt').value
    .split('\\n')
    .map(s => s.trim())
    .filter(Boolean);

  const normalize = document.getElementById('norm').checked;
  const n = parseInt(document.getElementById('n').value || "12", 10);

  const payload = { inputs: lines.length === 1 ? lines[0] : lines, normalize };

  document.getElementById('out').textContent = "Embedding...";
  document.getElementById('meta').textContent = "";

  const res = await fetch('/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  if (!res.ok) {
    document.getElementById('out').textContent = JSON.stringify(data, null, 2);
    return;
  }

  const emb = data.embeddings;
  const dim = emb[0]?.length ?? 0;

  document.getElementById('meta').textContent =
    `model: ${data.model} | inputs: ${emb.length} | dim: ${dim}`;

  const preview = emb.map((vec, i) => ({
    index: i,
    preview: vec.slice(0, n),
  }));

  document.getElementById('out').textContent = JSON.stringify(preview, null, 2);
}
</script>

</body>
</html>
"""

@app.on_event("startup")
def load_model():
    global model
    try:
        logger.info("Loading model: %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model.")
        raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}") from e


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    texts: List[str] = [req.inputs] if isinstance(req.inputs, str) else req.inputs

    try:
        vectors = model.encode(
            texts,
            normalize_embeddings=req.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = vectors.astype("float32").tolist()
        return EmbedResponse(embeddings=embeddings, model=MODEL_NAME)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Embedding generation failed.")
        raise HTTPException(status_code=500, detail="Embedding generation failed.") from e