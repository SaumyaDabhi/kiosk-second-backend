import os, json, re
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

JSONL_PATH = "disease_fulltext_common.jsonl"

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

raw_rows = load_jsonl(JSONL_PATH)
print(f"Loaded {len(raw_rows)} rows")

docs_raw: List[Document] = []
skipped = 0
for r in raw_rows:
    cond = r.get("title") or r.get("condition_name") or ""
    text = (r.get("text") or "").strip()
    if cond and text:
        docs_raw.append(Document(page_content=text, metadata={"condition": cond}))
    else:
        skipped += 1

print(f"Docs: {len(docs_raw)} | Skipped: {skipped}")
print("Example metadata:", docs_raw[0].metadata if docs_raw else "No docs")


# TODO: load your saved FAISS/BM25/reranker objects once here.
# from my_rag import retrieve_grouped_by_condition, format_query_bge, rerank, ensemble
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # ~300–600 tokens depending on tokenizer
    chunk_overlap=150,
    separators=["\n", ". ", " ", ""]
)

# Properly flatten chunks into a flat list of Document objects
docs = []
for d in docs_raw:
    for chunk in splitter.split_text(d.page_content):
        docs.append(Document(page_content=chunk, metadata=d.metadata))

BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "faiss_common_condition_index"
print("Loading FAISS index from:", FAISS_DIR)

faiss_store = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 50  # widen keyword recall

vec_retriever = faiss_store.as_retriever(search_kwargs={"k": 20})

ensemble = EnsembleRetriever(
    retrievers=[vec_retriever, bm25],
    weights=[0.5, 0.5]
)

#-------------------------------RE RANKER----------------------------------------
reranker_model_name = "BAAI/bge-reranker-base"   # good quality; use '...-small' for speed or '...-large' for best
cross_encoder = CrossEncoder(reranker_model_name, device=DEVICE)

TOP_CANDIDATES = 24  # rerank this many; tune for speed/quality

def rerank(query: str, candidates: List[Document]) -> List[Tuple[Document, float]]:
    pairs = [(query, d.page_content) for d in candidates]
    scores = cross_encoder.predict(pairs)  # higher is better
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked

#---------------------Retrieval → Rerank → Group-by-condition → Condition scores---------------------------

def retrieve_grouped_by_condition(
    query: str,
    ensemble_retriever: EnsembleRetriever,
    reranker_fn,
    top_candidates: int = TOP_CANDIDATES,
    top_conditions: int = 5,
    chunks_per_condition: int = 3,
) -> Dict[str, Dict]:
    # 1) Hybrid recall
    candidates = ensemble_retriever.invoke(query)
    # 2) Rerank
    ranked = reranker_fn(query, candidates)[:top_candidates]

    # 3) Group by condition
    by_cond: Dict[str, List[Tuple[Document, float]]] = {}
    for doc, score in ranked:
        cond = doc.metadata.get("condition", "Unknown")
        by_cond.setdefault(cond, []).append((doc, score))

    # 4) Aggregate per condition
    cond_scores = []
    for cond, items in by_cond.items():
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        top_items = items_sorted[:chunks_per_condition]
        base = sum(s for _, s in top_items)
        diversity_bonus = 0.05 * (len(top_items) - 1)  # small bonus for multiple evidence chunks
        cond_scores.append((cond, base + diversity_bonus, top_items))

    # 5) Pick top conditions
    cond_scores.sort(key=lambda x: x[1], reverse=True)
    chosen = cond_scores[:top_conditions]

    # 6) Pack result
    out = {}
    for cond, agg_score, items in chosen:
        out[cond] = {
            "condition_score": float(agg_score),
            "evidence": [
                {
                    "score": float(score),
                    "snippet": re.sub(r"\s+", " ", doc.page_content)[:600],
                }
                for doc, score in items
            ],
        }
    return out

def format_query_bge(q: str) -> str:
    # BGE query instruction helps retrieval
    return f"Represent this query for retrieving relevant documents: {q}"


app = FastAPI(title="Condition Triage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class RetrieveRequest(BaseModel):
    query: str
    k: int = 5

class RetrieveResponse(BaseModel):
    conditions: List[str]
    diagnostic: Dict[str, Any] = {}

class TriageRequest(BaseModel):
    query: str
    conditions: List[str]  # names only

class TriageResponse(BaseModel):
    triage_level: str
    primary_condition: str
    ranking: List[Dict[str, Any]]
    follow_up_actions: List[str]
    watch_outs: List[str]
    confidence: str

TRIAGE_RUBRIC = """
You are a medical triage helper. You are NOT a doctor and you do NOT diagnose. 

Input:
- Patient symptom description (query)
- A list of possible conditions (from a search system)

Your tasks:
1) For EACH condition, choose a likelihood band:
   very_unlikely, unlikely, possible, likely, very_likely
2) Choose ONE action level:
   self_treatment | wait_to_be_seen | go_to_clinic_next_open_day | go_to_er

Action level definitions (choose ONE):
- self_treatment:
  Mild symptoms. No danger signs. Safe simple steps are enough for now.
  Example: mild cold symptoms, mild headache without danger signs.

- wait_to_be_seen:
  Not urgent right now. Symptoms should be watched.
  Tell what to watch for and when to get help.
  Example: mild fever <24 hours, mild stomach upset without dehydration.

- go_to_clinic_next_open_day:
  Needs a medical check soon, but not emergency right now.
  Example: painful urination, wound that looks infected but person is stable, fever lasting multiple days.

- go_to_er:
  Any danger signs OR you cannot rule out a serious problem.
  Danger signs include: chest pain/pressure, trouble breathing, fainting, severe dizziness, new confusion,
  weakness on one side, trouble speaking, coughing blood, severe bleeding, severe dehydration, severe head injury.
  If unsure between clinic and ER, choose go_to_er.

Safety rule:
- If symptoms could be a serious heart, lung, brain, or bleeding problem, choose go_to_er even if the condition list
  does not include those diseases.
- Do NOT mention any medicines, pills, drugs, dosages, or brand names.
- Do NOT say “take aspirin”, “take ibuprofen”, “use antibiotics”, etc.
- Only suggest non-medicine actions like: rest, drink water, go to clinic/ER, call 911, ask staff for help.
- The patient you are helping are homeless who do not have access to many basic things so write your response accordingly by avoiding mentioning of house, place etc.

Language rules (VERY IMPORTANT):
- Write like you are talking to a 5th grader.
- Use short sentences.
- Avoid medical words. If you must use one, explain it simply.
- Assume the person may not have money, a phone, transportation, or a safe place to rest.
- Give practical steps that work in that situation.


Output rules:
- Return ONLY a JSON object that matches the schema given.
- Do not add extra keys.
- Do not include markdown.
"""

def call_gpt5_triage(query: str, condition_names: List[str]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": TRIAGE_RUBRIC.strip()},
        {"role": "user", "content": json.dumps({
            "task": "Estimate likelihoods and choose ONE action level for the patient.",
            "input": {"query": query, "conditions": condition_names},
            "output_schema": {
                "triage_level": "self_treatment | wait_to_be_seen | go_to_clinic_next_open_day | go_to_er",
                "level_explanation": "1-2 very short sentences explaining the chosen action level in simple words",
                "primary_condition": "one from the provided list",
                "ranking": [{
                    "condition": "string",
                    "likelihood_band": "very_unlikely | unlikely | possible | likely | very_likely",
                    "rationale": ["1-3 very short bullets in simple words"],
                    "red_flags_present": ["list"],
                    "missing_critical_info": ["list"]
                }],
                "follow_up_actions": ["2 very simple bullets the person can follow"],
                "watch_outs": ["2 very simple danger signs that mean go to ER/call 911"],
                "confidence": "low | medium | high"
            }
        }, ensure_ascii=False)}
    ]
    resp = client.chat.completions.create(
        model="gpt-5",
        response_format={"type": "json_object"},
        messages=messages
    )
    print(resp.choices[0].message.content)
    return json.loads(resp.choices[0].message.content)


@app.get("/")
def read_root():
    return {
        "message": "Triage backend is running.",
        "endpoints": ["/api/retrieve", "/api/triage", "/docs", "/redoc"],
    }

@app.post("/api/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    # ---- Replace this block with your actual retrieval call ----
    results = retrieve_grouped_by_condition(
        query=format_query_bge(req.query),
        ensemble_retriever=ensemble,
        reranker_fn=rerank,
        top_candidates=24,
        top_conditions=req.k,
        chunks_per_condition=3,
    )
    condition_names = list(results.keys())[:req.k]

    # optional: build a small diagnostic preview
    preview = [
        {
            "condition": cond,
            "condition_score": round(results[cond]["condition_score"], 3),
        }
        for cond in condition_names
    ]

    diagnostic = {
        "params": {
            "top_candidates": 24,
            "chunks_per_condition": 3,
            "requested_k": req.k,
        },
        "preview": preview,
    }

    return {
        "conditions": condition_names,
        "diagnostic": diagnostic,
    }
    # ------------------------------------------------------------
    # Temporary stub (until wired to your RAG objects):
    # condition_names = ["Hypertension", "Common cold", "Migraine"][:req.k]
    # diagnostic = {"note": "stub; replace with real retrieval outputs"}
    # return {"conditions": condition_names, "diagnostic": diagnostic}

@app.post("/api/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    data = call_gpt5_triage(req.query, req.conditions)
    return data
