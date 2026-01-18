import os, re
import numpy as np
from flask import Flask, render_template, request

try:
    import gensim.downloader as api
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

app = Flask(__name__, template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CORPUS_PATH = os.path.join(PROJECT_ROOT, "data", "corpus.txt")
TOPK = 10


MODEL_PATHS = {
    "Skipgram": os.path.join(PROJECT_ROOT, "models", "skipgram_softmax_w4.npz"),
    "Skipgram (NEG)": os.path.join(PROJECT_ROOT, "models", "skipgram_neg_w4.npz"),
    "GloVe": os.path.join(PROJECT_ROOT, "models", "glove_scratch_w4.npz")}

DEFAULT_MODEL = "Skipgram (NEG)"

with open(CORPUS_PATH, encoding="utf8") as f:
    RAW_CONTEXTS = [line.strip() for line in f if line.strip()]

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["emb"].astype(np.float32), list(d["vocabs"])

def build_w2i(vocabs):
    return {str(w).lower(): i for i, w in enumerate(vocabs)}

def tokenize_alpha(text: str):
    return [t.lower() for t in text.split() if t.isalpha()]

def text_to_vec(text, E, w2i):
    toks = tokenize_alpha(text)
    vecs = []
    for t in toks:
        idx = w2i.get(t)
        if idx is not None:
            vecs.append(E[idx])
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0).astype(np.float32)


CACHE = {}

def prepare_model_index(model_name: str):

    if model_name in CACHE:
        return CACHE[model_name]

    if model_name in MODEL_PATHS:
        path = MODEL_PATHS[model_name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        E, vocabs = load_npz(path)
        w2i = build_w2i(vocabs)

        contexts = []
        ctx_vecs = []
        for line in RAW_CONTEXTS:
            v = text_to_vec(line, E, w2i)
            if v is not None:
                contexts.append(line)
                ctx_vecs.append(v)

        C = np.stack(ctx_vecs).astype(np.float32)
        CACHE[model_name] = {"contexts": contexts, "C": C, "E": E, "w2i": w2i}
        return CACHE[model_name]

    if model_name == "GloVe (gensim)":
        if not HAS_GENSIM:
            raise RuntimeError("gensim is not available in this environment. Install gensim to use this model.")

        gensim_glove = api.load("glove-wiki-gigaword-100")
        corpus_words = sorted({w for line in RAW_CONTEXTS for w in tokenize_alpha(line)})
        words_in = [w for w in corpus_words if w in gensim_glove.key_to_index]
        w2i = {w: i for i, w in enumerate(words_in)}
        E = np.stack([gensim_glove[w] for w in words_in]).astype(np.float32)

        contexts = []
        ctx_vecs = []
        for line in RAW_CONTEXTS:
            toks = tokenize_alpha(line)
            vecs = []
            for t in toks:
                idx = w2i.get(t)
                if idx is not None:
                    vecs.append(E[idx])
            if not vecs:
                continue
            contexts.append(line)
            ctx_vecs.append(np.mean(np.stack(vecs), axis=0).astype(np.float32))

        C = np.stack(ctx_vecs).astype(np.float32)
        CACHE[model_name] = {"contexts": contexts, "C": C, "E": E, "w2i": w2i}
        return CACHE[model_name]

    raise ValueError(f"Unknown model: {model_name}")


def search(query: str, model_name: str, topk: int = 10):
    idx = prepare_model_index(model_name)

    qv = text_to_vec(query, idx["E"], idx["w2i"])
    if qv is None:
        return [], "Query has no in-vocab tokens for the selected model."

    scores = idx["C"] @ qv
    top_idx = np.argsort(-scores)[:topk]

    results = [{"text": idx["contexts"][i], "score": float(scores[i])} for i in top_idx]
    return results, None

# Web
@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    error = None

    model_name = DEFAULT_MODEL

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        model_name = request.form.get("model", DEFAULT_MODEL)

        try:
            results, error = search(query, model_name, topk=TOPK)
        except Exception as e:
            results = []
            error = str(e)

    # build list for dropdown
    model_options = list(MODEL_PATHS.keys())
    if HAS_GENSIM:
        model_options.append("GloVe (gensim)")

    return render_template(
        "index.html",
        query=query,
        results=results,
        error=error,
        model=model_name,
        mode="Dot Product",
        model_options=model_options
    )


if __name__ == "__main__":
    app.run(debug=True)

