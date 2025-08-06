# O modelo inteiro é salvo em RAM.
# Se não houver RAM suficiente, o modelo pode rodar extremamente lento por causa de swap
# (acesso ao disco) ou falhar completamente com erro de alocação.

# Talvez só usar a IA para buscar contexto e retornar para o usuário, e não para gerar texto.

from langchain_community.llms.llamacpp import LlamaCpp
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
    "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    # 6
    "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    # 7
    "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    # 8
    "Gemma-3-Gaia-PT-BR-4b-it-q4_k_m.gguf",
    # 9
    "bode-7b-alpaca-q4_k_m.gguf"
]

MODEL = MODELS[9]
GGUF_PATH = f"./models/{MODEL}"
EMBED_MODEL = "./models/intfloat--multilingual-e5-small"
CHROMA_PATH = "./chroma_db"
CONTEXT_LIMIT = 3

DOCUMENTS = [
    "Nesse software há 3 tipos de alertas: fogo, falha e supervisão.",
    "Um dispositivo recebe o alerta de sem sinal quando fica mais de 5 minutos sem comunicação com o coordenador. Quando um dispositivo está sem sinal, é preciso tirar da tomada por um tempo e depois reconectá-lo. O recomendado é no mínimo 10 minutos fora da tomada. Se isso não funcionar, é necessário reiniciar o coordenador. A última opção seria contatar o suporte para investigar melhor o problema.",
    "O alerta de sem sinal pode ser resolvido reiniciando o dispositivo. Dessa forma, quando ele ligar novamente, ele forçará comunicação com o coordenador. Se o problema persistir, é necessário verificar se o dispositivo está conectado corretamente à energia e se a rede está funcionando.",
    "Um dispositivo recebe o alerta de bateria fraca quando o nível de bateria está abaixo de 10%. Quando um dispositivo está com bateria fraca, é necessário conectá-lo a energia imediatamente, isso fará com que ele recarregue. Se o dispositivo não for recarregado, ele pode parar de funcionar. O recomendado é recarregar o dispositivo por pelo menos 1 hora.",
    "Um dispositivo recebe o alerta de falta de energia quando ele não está recebendo energia elétrica. Quando um dispositivo está sem energia, é necessário verificar se o cabo de energia está conectado corretamente e se a tomada está funcionando. Se o problema persistir, pode ser necessário trocar o cabo de energia ou verificar a fiação elétrica.",
    "Existem 2 tipos de dispositivos: entrada e saída. Dispositivos de entrada são aqueles que enviam informações para o sistema, como sensores de fumaça e botões de pânico. Dispositivos de saída são aqueles que recebem informações do sistema, como sirenes e luzes de alerta. Exemplo de dispositivo de entrada: detector de fumaça. Exemplo de dispositivo de saída: audiovisual.",
]

embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if not os.path.exists(CHROMA_PATH) or len(os.listdir(CHROMA_PATH)) == 0:
    u.log_info("Indexando documentos...") 
    docs = [Document(page_content=doc) for doc in DOCUMENTS]
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory=CHROMA_PATH)
    u.log_info("Documentos indexados.")
else:
    u.log_info("Documentos já indexados.")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

stream_handler = StreamingStdOutCallbackHandler()

llm = LlamaCpp(
    model_path=GGUF_PATH,
    n_ctx=3072,
    n_threads=2,
    max_tokens=256,
    verbose=False,
    streaming=True,
    # callbacks=[stream_handler],
    temperature=0.5,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate.from_template("""
Você é um assistente especializado em responder com base *apenas* no contexto fornecido. Ignore perguntas que não puder responder com base no contexto (responda com: "Não tenho informações sobre isso"). Sempre responda em português do Brasil.

Contexto:
{context}

Histórico da conversa:
{chat_history}

Pergunta:
{question}

Resposta:
""")

retriever = db.as_retriever(kwargs={"k": CONTEXT_LIMIT})

def stream_llm(input_str: str):
    start = time.perf_counter()

    response = ""
    first_token_time = None

    for token in llm.stream(input_str):
        if first_token_time is None:
            first_token_time = time.perf_counter()

        response += token

    end = time.perf_counter()

    u.log_info(f"Primeiro token recebido após {first_token_time - start:.2f} segundos")
    u.log_info(f"Tempo de resposta: {end - start:.2f} segundos")

    print(u.COLOR_ANSWER + response + u.RESET)

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

def interactive_mode():
    while True:
        query = input(f"\n{u.COLOR_QUESTION}Pergunta: {u.RESET}")
        answer = rag_chain.invoke(query)

        memory.save_context({"input": query}, {"output": answer})

def script_mode():
    queries = [
        "Quais são os tipos de alertas no software?",
        "O que fazer quando um dispositivo está sem sinal?",
        "Quantos gols um time precisa para vencer uma partida de futebol?",
        "Quantas patas tem um cão?",
        "Qual é a capital da França?"
    ]

    for query in queries:
        print(f"\n{u.COLOR_QUESTION}Pergunta: {query}{u.RESET}")
        answer = rag_chain.invoke(query)

        memory.save_context({"input": query}, {"output": answer})

if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "script":
        script_mode()
    else:
        interactive_mode()
