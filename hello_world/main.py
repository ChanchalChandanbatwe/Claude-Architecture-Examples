import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage
from sdk_parser import log_message

print("Starting agentic loop...")
async def main():
    # Agentic loop: streams messages as Claude works
    async for message in query(
        prompt="Change hello_world.rb to print 'Hello, World!' with a comma and capital W.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob"],  # Auto-approve these tools
            permission_mode="acceptEdits",  # Auto-approve file edits
        ),
    ):
        # Log every message to hello_world/logs/<timestamp>.log
        log_message(message)

        # Print human-readable output
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)  # Claude's reasoning
                elif hasattr(block, "name"):
                    print(f"Tool: {block.name}")  # Tool being called
        elif isinstance(message, ResultMessage):
            print(f"Done: {message.subtype}")  # Final result


asyncio.run(main())