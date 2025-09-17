import os

print("Key present?", bool(os.getenv("GROQ_API_KEY")))
print("Value (first 8 chars):", os.getenv("GROQ_API_KEY")[:8] + "..." if os.getenv("GROQ_API_KEY") else None)
