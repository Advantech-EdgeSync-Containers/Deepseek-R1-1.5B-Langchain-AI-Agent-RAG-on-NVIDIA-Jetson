import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uuid
import os
import asyncio
import requests

from utils import ChatTokenUtils
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains import RetrievalQA
from llm_loader import get_llm
from rag_utils import load_all_pdfs, split_docs, create_vectorstore
from schema import ChatRequest

# Global retriever (PDF loaded once)
retriever = None

# ---- FastAPI App
app = FastAPI()
TIMEOUT_SECONDS = int(os.getenv("MAX_PROMPT_GENERATION_TIMEOUT_IN_MIN", 20)) * 60


# ---- Preload PDF at Startup
@app.on_event("startup")
async def load_rag_resources():
    global retriever
    print("[INFO] Application is starting...",flush=True)
    print("[INFO] Please wait while the PDF is being loaded and the embedding model is initialized.",flush=True)

    #Load all the documents in dir
    docs = load_all_pdfs('pdfs')
    chunks = split_docs(docs)

    print("[INFO] Embedding chunks and building FAISS index...",flush=True)
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": int(os.getenv("K","1")), "score_threshold": float(os.getenv("SCORE_THRESHOLD","0.6"))})

    print("[INFO] RAG setup complete. Model is ready to serve queries.\n",flush=True)

# ---- Token Streaming
async def token_stream(agent, model_name, prompt, callback, request: Request):
    task = asyncio.create_task(agent.ainvoke(prompt))
    try:
        start_time = time.time()
        async for token in callback.aiter():
            if await request.is_disconnected():
                print("Client disconnected.")
                task.cancel()
                break
            if token:
                if time.time() - start_time > TIMEOUT_SECONDS:
                    print("Stream Chat Timeout: Stopping stream...")
                    task.cancel()
                    agent.stop(model_name)
                    break
                token = token.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                yield f'data: {{"id": "{uuid.uuid4()}", "object": "chat.completion.chunk", "choices": [{{"delta": {{"content": "{token}"}}, "index": 0, "finish_reason": null}}]}}\n\n'

        if not task.done():
            await task
        yield 'data: [DONE]\n\n'

    except asyncio.CancelledError:
        task.cancel()
        print("Streaming task cancelled.")
    except Exception as e:
        print(f"Streaming error: {e}")
        yield 'data: [DONE]\n\n'

# ---- Chat Completion Endpoint
@app.post("/chat/completions")
async def chat_completion(request: Request, chat_request: ChatRequest):
    global retriever

    prompt = chat_request.messages[-1].content
    model_name = chat_request.model

    # Prepare RAG
    callback = AsyncIteratorCallbackHandler()
    llm = get_llm(callback)
    rag_chain = RetrievalQA.from_chain_type(llm=llm,    chain_type="stuff",
 retriever=retriever)

    if chat_request.stream:
        return StreamingResponse(
            token_stream(rag_chain, model_name, prompt, callback, request),
            media_type="text/event-stream"
        )
    else:
        try:
            result = await asyncio.wait_for(llm.ainvoke(prompt), timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            print("No Stream LLM call timed out.")
            # Optionally stop the model if still running
            result = "The request took too long and was stopped."
        return ChatTokenUtils.build_chat_response(result, model_name)

# ---- Model Listing for OpenWebUI
@app.get("/models")
async def list_models():
    try:
        response = requests.get(f"{os.getenv('OLLAMA_API_BASE')}/api/tags")
        response.raise_for_status()
        tags = response.json().get("models", [])
    except Exception as e:
        return {"error": f"Failed to fetch models: {str(e)}"}

    return {
        "object": "list",
        "data": [{
            "id": model["name"],
            "object": "model",
            "size": model.get("size"),
            "modified": model.get("modified_at"),
            "owned_by": "user"
        } for model in tags]
    }
