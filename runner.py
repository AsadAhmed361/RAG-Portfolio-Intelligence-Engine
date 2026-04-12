import asyncio
import sys
from rag_engine import PortfolioRetriever, DocumentChunker, EmbedEngine, SearchEngine, ChatEngine


async def run_update():
    retriever = PortfolioRetriever()
    return await retriever.run_update()


def run_chunk():
    chunker = DocumentChunker(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = chunker.process_pdf("Asad_Ahmed_Professional_RAG.pdf")

    chunker.save_chunks(chunks, "chunks.json")

    chunker.preview(chunks)

    print(f"\nTotal Chunks: {len(chunks)}")

def run_embed():
    """
    Embedding pipeline ko execute karne wala function
    """
    print("Starting Embedding Process...")
    engine = EmbedEngine()
    success = engine.generate_and_save(
        input_path="chunks.json", 
        output_path="embedded_chunks.json"
    )
    
    if success:
        print("Embedding step completed successfully.")
    else:
        print("Embedding step failed.")

async def run_full():
    print("Starting full pipeline...\n")

    # Step 1: Web Crawl & PDF Gen
    success = await run_update()
    if not success:
        print("AI update failed, stopping pipeline")
        return

    # Step 2: PDF Chunking
    run_chunk()

    # Step 3: Vector Embedding
    run_embed()

    print("\nFull Pipeline completed successfully (Crawl -> PDF -> Chunk -> Embed)")

def run_test_search():
    searcher = SearchEngine()
    query = "What are Asad's expertise in low-level programming?"
    
    # Ye function search bhi karega aur file bhi save karega
    results = searcher.get_top_matches(query, top_k=3)
    
    print(f"\nSearch complete. Top score: {results[0]['score']:.4f}")
    

def run_rag_flow(user_query):
    # STEP 1: Search (Updates search_results.json)
    searcher = SearchEngine()
    searcher.get_top_matches(user_query, top_k=3)
    
    # STEP 2: Ask (Reads the JSON and calls Gemini)
    chat = ChatEngine()
    answer = chat.generate_response()
    
    print(f"\n🤖 ASAD AI: {answer}")

def show_help():
    print("""
Available commands:

python runner.py update  -> AI + PDF generation
python runner.py chunk   -> Only chunk PDF to JSON
python runner.py embed   -> Only generate vectors from JSON
python runner.py search "query" -> Search and save results to JSON
python runner.py ask     -> Generate answer using existing search_results.json
python runner.py full    -> Full pipeline (Update + Chunk + Embed)
""")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "update":
        asyncio.run(run_update())

    elif command == "chunk":
        run_chunk()

    elif command == "embed":
        run_embed()

    elif command == "search":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What are Asad's skills in low-level programming?"
        run_search(query)

    elif command == "ask":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What are Asad's skills in low-level programming?"
        run_rag_flow(query)

    elif command == "full":
        asyncio.run(run_full())

    else:
        show_help()