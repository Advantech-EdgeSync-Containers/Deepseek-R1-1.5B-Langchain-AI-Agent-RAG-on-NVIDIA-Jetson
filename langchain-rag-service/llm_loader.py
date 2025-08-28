import os
from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler


def to_int(value):
    return int(value) if value and value.isdigit() else None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_llm(callback_handler=None):
    system_prompt = """
         You are a helpful assistant with access to a knowledge base (document) and general world knowledge.

        - If the question is general (like about greeting(Hi, Hello),science, history, math, biology, chemistry etc.), answer using your own knowledge.
        - If the question is specific to the document uploaded(e.g., EdgeSync, containers, APIs or anything which you can be a reference to document), refer to the document context provided.
        - If the document is not relevant to the question, respond using your own knowledge.
        - Never use the document for greetings like Hi, Hello 

        ## 📌 EXAMPLES
        User: Hi or Hello?
        → Hi, How can I assist you ?
    """

    class MyLoggingHandler(BaseCallbackHandler):
        def on_llm_start(self, *args, **kwargs):
            print("\n=== LLM Request Log Start ===")

        def on_llm_end(self, *args, **kwargs):
            print("\n=== LLM Request Log End ===")

    return Ollama(
        base_url=os.getenv("OLLAMA_API_BASE"),
        model=os.getenv("MODEL_NAME"),
        temperature=to_float(os.getenv("TEMPERATURE", 0.7)),
        verbose=True,
        num_ctx=to_int(os.getenv("NUM_CTX")),
        num_gpu=to_int(os.getenv("NUM_GPU")),
        num_thread=to_int(os.getenv("NUM_THREAD")),
        system=os.getenv("SYSTEM",system_prompt),
        keep_alive=os.getenv("KEEP_ALIVE"),
        repeat_penalty=to_float(os.getenv("REPEAT_PENALTY")),
        template=os.getenv("TEMPLATE"),
        top_p=to_float(os.getenv("TOP_P")),
        top_k=to_int(os.getenv("TOP_K")),
        callbacks=(
            [callback_handler, MyLoggingHandler()]
            if callback_handler
            else [MyLoggingHandler()]
        ),
    )
