import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")
warnings.filterwarnings("ignore", category=UserWarning, message=".*max_new_tokens.*max_length.*")

import asyncio
from llm import LocalLLM
from router import ToolRouter
from mcp_client import MCPClient

# MODEL_PATH = "outputs-full/checkpoint-132"
MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


router = ToolRouter()
llm = LocalLLM(MODEL_PATH)

mcp = MCPClient(
    command="python",
    args=["mcp_server.py"]  # change this
)


async def run():
    async with mcp.run() as (read, write):
        async with __import__("mcp").ClientSession(read, write) as session:
            await session.initialize()

            print("\nAgent ready. Type 'exit' to quit.\n")

            while True:
                user_input = input("You: ").strip()

                if user_input in ["exit", "quit"]:
                    break

                if not user_input:
                    continue

                # --------------------
                # TOOL ROUTING (ML)
                # --------------------
                decision = router.route(user_input)

                if decision:
                    print(f"\n[Tool: {decision['tool']} | {decision['confidence']:.2f}]")

                    # Call the tool
                    result = await session.call_tool(
                        decision["tool"],
                        {"query": user_input}
                    )

                    # # Extract and display raw data
                    # tool_data = str(result)
                    # if "TextContent" in tool_data and "text='" in tool_data:
                    #     import re
                    #     match = re.search(r"text='([^']*)'", tool_data.replace("\\n", "\n"))
                    #     if match:
                    #         tool_data = match.group(1)
                    
                    # print(f"\n{tool_data}\n")

                    # --- LLM FORMATTING ---
                    # Format the tool result for the LLM to present naturally
                    tool_data = str(result)
                    
                    # Build a prompt for the LLM
                    llm_prompt = [
                        {
                            "role": "user",
                            "content": f"Question: {user_input}\n\nI retrievedd this data:\n\n{tool_data}\n\nPlease present this information in a friendly, conversational way as if answering the question directly."
                        }
                    ]

                    # Generate natural response from LLM
                    response = llm.generate(llm_prompt)
                    print("Assistant:", response)

                    continue

                # --------------------
                # NORMAL LLM PATH
                # --------------------
                response = llm.generate([{"role": "user", "content": user_input}])

                print("Assistant:", response)


if __name__ == "__main__":
    asyncio.run(run())