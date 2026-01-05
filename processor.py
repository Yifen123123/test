import requests
import chardet


OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"


def load_text_any_encoding(file_path: str) -> str:
    raw = open(file_path, "rb").read()
    guess = chardet.detect(raw)
    enc = guess.get("encoding") or "utf-8"
    try:
        return raw.decode(enc)
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def call_ollama(prompt: str) -> str:
    resp = requests.post(OLLAMA_API_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }, timeout=180)

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API 回應錯誤: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get("response", "")


def process_file(file_path: str, prompt_path: str):
    try:
        transcript = load_text_any_encoding(file_path)
        system_prompt = load_prompt_template(prompt_path)

        final_prompt = f"""{system_prompt}

=== Transcript ===
{transcript}
"""

        output = call_ollama(final_prompt)
        return output.strip()

    except Exception as e:
        return {"error": str(e)}
