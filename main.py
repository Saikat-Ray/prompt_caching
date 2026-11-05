import os
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080")
r = client.messages.create(model="claude-haiku-5-5", max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in five words."}])