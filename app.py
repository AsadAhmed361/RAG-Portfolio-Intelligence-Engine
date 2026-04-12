from fastapi import FastAPI
from rag_engine import SearchEngine, ChatEngine

app = FastAPI()

# Model sirf aik baar load hoga server start hote waqt
print("🚀 Loading Model into RAM...")
searcher = SearchEngine()
chat = ChatEngine()

@app.get("/ask")
async def ask_asad(query: str):
    # Zero loading time here!
    results = searcher.get_top_matches(query)
    answer = chat.generate_response()
    return {"answer": answer}