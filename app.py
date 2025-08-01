from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import util as u

# --- Config ---
# GGUF_PATH = "./models/Mistral-7B-Instruct-v0.1.Q5_K_M.gguf"
# GGUF_PATH = "./models/Mistral-7B-Instruct-v0.3.Q5_K_M.gguf"
GGUF_PATH = "./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
EMBED_MODEL = "./models/intfloat--multilingual-e5-small"
CONTEXT_LIMIT = 3

DOCUMENTS = [
  "Nesse software há 3 tipos de alertas: fogo, falha e supervisão.",
  "Em uma partida de futebol, o time com mais gols vence.",
  "Cães têm quatro patas e são conhecidos por sua lealdade.",
  "A capital da França é Berlim."
]

# --- Init ---
u.log_info("Carregando modelo de embeddings...")
embedder = SentenceTransformer(EMBED_MODEL)

u.log_info("Carregando base vetorial (Chroma)...")
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="evo")

u.log_info("Indexando documentos...")
for i, doc in enumerate(DOCUMENTS):
  emb = embedder.encode(doc).tolist()
  collection.add(documents=[doc], embeddings=[emb], ids=[f"doc-{i}"])

u.log_info("Carregando modelo LLaMa...")
llm = Llama(
  model_path=GGUF_PATH,
  n_ctx=2048,
  n_threads=8,
  verbose=False
)

def answer_query(query):
  emb_query = embedder.encode(query).tolist()
  result = collection.query(query_embeddings=[emb_query], n_results=CONTEXT_LIMIT)

  context = "\n".join(result['documents'][0])

  prompt = f"""
    Você é um assistente inteligente que responde perguntas com base em um contexto fornecido. Responda de forma clara e concisa.
    Não inclua informações irrelevantes ou suposições. Se não souber a resposta, diga que não sabe. Os usuários falam com você através de um sistema de alarme de incêndios da empresa GlobalSonic. Esse sistema é responsável por monitorar e alertar sobre incêndios, além de gerenciar os dispositivos.

    Contexto: {context}

    Pergunta: {query}

    Resposta:
  """

  u.log_info("\nContexto recuperado:")
  print(context)

  u.log_info("\nGerando resposta com LLaMa...")

  response = llm(prompt, max_tokens=256, stream=True)

  u.log_info("\nResposta gerada pelo modelo LLaMa:")

  for chunk in response:
      print(u.COLOR_ANSWER + chunk["choices"][0]["text"] + u.RESET, end="", flush=True)

  print("\n")

if __name__ == "__main__":
  while True:
    query = input(f"\n{u.COLOR_QUESTION}> Digite sua pergunta: {u.RESET}")
    answer_query(query)