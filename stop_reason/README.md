# stop_reason example

Run it and observe the output:

```bash
pip install claude-agent-sdk
python stop_reason/main.py
```

Uses the built-in `WebFetch` tool against the public `wttr.in` weather endpoint
(no API key, no custom tool or MCP server). Each assistant turn prints its
`stop_reason` (`tool_use`, then `end_turn`), followed by the tool result and the
final `[Done]` line from `lib/sdk_parser`.
