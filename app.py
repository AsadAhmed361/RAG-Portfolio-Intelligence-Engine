#uvicorn app:app --reload
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from rag_engine import PortfolioRetriever, DocumentChunker, EmbedEngine, SearchEngine, ChatEngine
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uuid
from collections import defaultdict
chat_sessions = defaultdict(list)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


temp_storage = {
    "raw_chunks": [],
    "embedded_data": []
}

search_engine = SearchEngine()
chat_engine = ChatEngine()

session_store = {}

def get_session(session_id: str | None):
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in session_store:
        session_store[session_id] = []

    return session_id, session_store[session_id]

async def full_sync_pipeline():
    try:
        logger.info("Starting Full Sync: Crawling latest data...")

        retriever = PortfolioRetriever()
        await retriever.run_update()

        logger.info("Syncing: Chunking updated PDF...")
        chunker = DocumentChunker()
        temp_storage["raw_chunks"] = chunker.process_pdf("Asad_Ahmed_Master_RAG.pdf")

        logger.info("Syncing: Generating new vectors...")
        embedder = EmbedEngine()
        temp_storage["embedded_data"] = embedder.generate_vectors(temp_storage["raw_chunks"])

        search_engine.update_index(temp_storage["embedded_data"])

        logger.info("Full Sync Complete: RAM is now updated with latest data.")

    except Exception as e:
        logger.error(f"Sync Pipeline Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initial startup load from existing PDF...")

    try:
        chunker = DocumentChunker()
        temp_storage["raw_chunks"] = chunker.process_pdf("Asad_Ahmed_Master_RAG.pdf")

        embedder = EmbedEngine()
        temp_storage["embedded_data"] = embedder.generate_vectors(temp_storage["raw_chunks"])

        search_engine.update_index(temp_storage["embedded_data"])

    except Exception as e:
        logger.warning(f"Initial load failed (Normal if PDF doesn't exist yet): {e}")

    yield


app = FastAPI(title="Pro-RAG Master Sync API", lifespan=lifespan)


@app.post("/pipeline/update")
async def trigger_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(full_sync_pipeline)

    return {"message": "Master update triggered. Crawling, chunking, and embedding are running in background."}


chat_history_buffer = []


@app.get("/ask")
async def ask_asad(query: str):
    try:
        results = search_engine.get_top_matches(query)

        answer = chat_engine.generate_response(
            query=query,
            search_results=results,
            history=chat_history_buffer
        )

        chat_history_buffer.append({"user": query, "assistant": answer})

        if len(chat_history_buffer) > 5:
            chat_history_buffer.pop(0)

        return {"query": query, "answer": answer}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return FileResponse("index.html")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.message
        session_id = request.session_id

        # get session history
        history = chat_sessions[session_id]

        results = search_engine.get_top_matches(query)

        answer = chat_engine.generate_response(
            query=query,
            search_results=results,
            history=history
        )

        # store per session
        history.append({
            "user": query,
            "assistant": answer
        })

        # optional limit (last 10 turns)
        if len(history) > 5:
            history.pop(0)

        chat_sessions[session_id] = history

        return {"reply": answer}

    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return {"reply": "Sorry, I am having trouble connecting to my brain right now."}