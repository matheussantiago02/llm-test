from langchain_community.llms.llamacpp import LlamaCpp
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import util as u
import os
import time

MODELS = [
    # 0
    "AV-BI-Qwen2.5-3B-PT-BR-Instruct.Q4_K_M.gguf",
    # 1
    "AV-BI-Qwen2.5-3B-PT-BR-Instruct.Q6_K.gguf",
    # 2
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    # 3
    "Mistral-7B-Instruct-v0.1.Q5_K_M.gguf",
    # 4
    "Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
    # 5
    "Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
    # 6
    "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    # 7
    "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    # 8
    "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    # 9
    "Gemma-3-Gaia-PT-BR-4b-it-q4_k_m.gguf"
]

MODEL = MODELS[9]
GGUF_PATH = f"./models/{MODEL}"
EMBED_MODEL = "./models/intfloat--multilingual-e5-small"
CHROMA_PATH = "./chroma_db"
CONTEXT_LIMIT = 3

DOCUMENTS = [
    "Nesse software há 3 tipos de alertas: fogo, falha e supervisão.",
    "Um dispositivo recebe o alerta de sem sinal quando fica mais de 5 minutos sem comunicação com o coordenador. Quando um dispositivo está sem sinal, é preciso tirar da tomada por um tempo e depois reconectá-lo. O recomendado é no mínimo 10 minutos fora da tomada. Se isso não funcionar, é necessário reiniciar o coordenador. A última opção seria contatar o suporte para investigar melhor o problema."
    "Em uma partida de futebol, o time com mais gols vence.",
    "Cães têm quatro patas e são conhecidos por sua lealdade.",
    "A capital da França é Berlim."
]

embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if not os.path.exists(CHROMA_PATH) or len(os.listdir(CHROMA_PATH)) == 0:
    u.log_info("Indexando documentos...") 
    docs = [Document(page_content=doc) for doc in DOCUMENTS]
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory=CHROMA_PATH)
    db.persist()
    u.log_info("Documentos indexados.")
else:
    u.log_info("Documentos já indexados.")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

stream_handler = StreamingStdOutCallbackHandler()

llm = LlamaCpp(
    model_path=GGUF_PATH,
    n_ctx=8192,
    n_threads=1,
    max_tokens=256,
    verbose=False,
    streaming=True,
    callbacks=[stream_handler],
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate.from_template("""
Você é um assistente útil que responde com base no contexto fornecido. Sempre responda em português do Brasil.

Contexto:
{context}

Histórico da conversa:
{chat_history}

Pergunta:
{question}

Resposta:
""")

retriever = db.as_retriever(kwargs={"k": CONTEXT_LIMIT})

def stream_llm(input_str):
    start = time.perf_counter()

    response = ""

    for token in llm.stream(input_str):
        response += token

    end = time.perf_counter()
    u.log_info(f"Tempo de resposta: {end - start:.2f} segundos")

    return response

rag_chain = (
    RunnableMap({
        "context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
        "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
    })
    | prompt
    | RunnableLambda(stream_llm)
)

while True:
    query = input(f"\n{u.COLOR_QUESTION}Pergunta: {u.RESET}")
    answer = rag_chain.invoke(query)

    memory.save_context({"input": query}, {"output": answer})
