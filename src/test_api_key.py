from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[{"role": "user", "content": "Reply with exactly: API works"}]
)

print(resp.content[0].text)
print("input tokens:", resp.usage.input_tokens)
print("output tokens:", resp.usage.output_tokens)