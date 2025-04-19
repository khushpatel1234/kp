import os
import json
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

base_model_path = "/content/drive/MyDrive/llama-2-7b-hf"
offload_dir = "/content/offload"
os.makedirs(offload_dir, exist_ok=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=offload_dir
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

model_paths = {
    "finance": "/content/drive/MyDrive/lora_finetuned_model_finance",
    "healthcare": "/content/drive/MyDrive/llama_finetuned_health",
    "technology": "/content/drive/MyDrive/lora_finetuned_model_tech"
}

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cpu")
SERPER_API_KEY = "your-serper-api-key"
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

memory = load_memory()

def perform_rag(state):
    query = state["query"]
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    data = json.dumps({"q": query, "num": 3})
    response = requests.post(url, headers=headers, data=data)
    context = ""
    if response.status_code == 200:
        results = response.json()
        context = "\n".join([f"{r['title']}: {r['link']}" for r in results.get("organic", [])])
    state["context"] = context
    return state

def detect_domains(state):
    query = state["query"]
    labels = ["finance", "healthcare", "technology"]
    result = classifier(query, candidate_labels=labels, multi_label=True)
    domains = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.3]
    state["domains"] = domains
    return state

def domain_router(state):
    return state

def similarity_score(query, response):
    query_emb = sim_model.encode(query, convert_to_tensor=True)
    response_emb = sim_model.encode(response, convert_to_tensor=True)
    return util.cos_sim(query_emb, response_emb).item()

def generate_answer(state):
    query = state["query"]
    context = state.get("context", "")
    history_context = "\n".join([f"Q: {item['query']}\nA: {item['response']}" for item in memory[-3:]])
    responses = []
    for domain in state.get("domains", []):
        model = PeftModel.from_pretrained(base_model, model_paths[domain], device_map="auto")
        prompt = f"Previous Context:\n{history_context}\n\nCurrent Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sim_score = similarity_score(query, response)
        responses.append((domain, response, sim_score))
        memory.append({"query": query, "response": response})
    save_memory(memory)
    state["responses"] = responses
    return state

def confidence_filter(state):
    final_responses = []
    responses = state.get("responses", [])
    if responses:
        for domain, response, score in responses:
            if score > 0.5:
                final_responses.append(f"{domain.title()} Answer: {response}")
            else:
                final_responses.append(f"{domain.title()} Answer is unclear. Try rephrasing.")
        state["final_response"] = "\n".join(final_responses)
    else:
        state["final_response"] = "No responses generated."
    return state

def output_node(state):
    state["final_response"] = state.get("final_response", "No final response produced.")
    return state
