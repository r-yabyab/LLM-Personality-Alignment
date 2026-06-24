from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self, command="python", args=None):
        self.server_params = StdioServerParameters(
            command=command,
            args=args or []
        )

    def run(self):
        return stdio_client(self.server_params)

    async def call_tool(self, session, name, args):
        return await session.call_tool(name, args)