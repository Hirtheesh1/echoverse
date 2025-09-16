# create hf_inference_test.py with the correct contents

from dotenv import load_dotenv
load_dotenv()
import os, traceback, requests
from huggingface_hub import HfApi
try:
    from huggingface_hub import InferenceClient
    INFERENCE_CLIENT_AVAILABLE = True
except Exception:
    InFERENCE_CLIENT_AVAILABLE = False
    InferenceClient = None

token = os.getenv("HF_TOKEN")
model = os.getenv("HF_API_MODEL", "tiiuae/falcon-7b-instruct")

print("HF_TOKEN present:", bool(token))
print("HF_API_MODEL:", model)
print("inference client available:", INFERENCE_CLIENT_AVAILABLE)

# whoami + model_info
try:
    api = HfApi()
    who = api.whoami(token=token)
    print("whoami ok:", who.get("name", who.get("id", "<unknown>")))
except Exception as e:
    print("whoami failed:", repr(e))

try:
    info = api.model_info(model, token=token)
    print("model_info ok:", getattr(info, "id", str(info)))
except Exception as e:
    print("model_info failed:", repr(e))

# Try InferenceClient if available
if INFERENCE_CLIENT_AVAILABLE:
    try:
        client = InferenceClient(token=token)
        try:
            r = client.text_generation(model=model, inputs="Rewrite: Atoms can split.", max_new_tokens=30, temperature=0.2)
        except TypeError:
            r = client.text_generation(model=model, prompt="Rewrite: Atoms can split.", max_new_tokens=30, temperature=0.2)
        print("InferenceClient result (truncated):", str(r)[:1000])
    except Exception as e:
        print("InferenceClient exception:", repr(e))
        traceback.print_exc()
else:
    print("InferenceClient not available; skipping client test.")

# Raw HTTP test against gpt2 (public model) to check endpoint/token
try:
    resp = requests.post(
        "https://api-inference.huggingface.co/models/gpt2",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": "Hello there"},
        timeout=30
    )
    print("gpt2 HTTP status:", resp.status_code)
    print("gpt2 HTTP body (truncated):", resp.text[:500])
except Exception as e:
    print("gpt2 HTTP error:", repr(e))
