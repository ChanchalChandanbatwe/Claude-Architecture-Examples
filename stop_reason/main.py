import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from claude_agent_sdk import ClaudeAgentOptions, query
from sdk_parser import log_message

options = ClaudeAgentOptions(
    model="claude-haiku-4-5-20251001",
    allowed_tools=["WebFetch"],
    max_turns=5,
)

PROMPT = "What's the weather in Paris? Fetch https://wttr.in/Paris?format=3"


async def main() -> None:
    async for message in query(prompt=PROMPT, options=options):
        print(log_message(message))
        if type(message).__name__ == "AssistantMessage":
            print(f"[stop_reason] {message.stop_reason}")


asyncio.run(main())
