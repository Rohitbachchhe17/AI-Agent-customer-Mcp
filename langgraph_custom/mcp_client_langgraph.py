import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

async def main():

    # Get OpenRouter API Key (stored in OPENAI_API_KEY)
    openrouter_key = os.getenv("OPENAI_API_KEY")

    # ✅ Initialize model using OpenRouter
    openrouter_api_key = "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"
    model = ChatOpenAI(
        model="openai/gpt-4o-mini",   # IMPORTANT
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # ✅ MCP Client (HTTP server)
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "streamable-http",
                "url": "http://127.0.0.1:8000/mcp"
            }
        }
    )

    # Load tools from MCP server
    tools = await client.get_tools()

    # Bind tools with model
    model_with_tools = model.bind_tools(tools)

    # Tool execution node
    tool_node = ToolNode(tools)

    # Decide whether to call tool or end
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Call LLM
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Build LangGraph
    builder = StateGraph(MessagesState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")

    builder.add_conditional_edges(
        "call_model",
        should_continue,
    )

    builder.add_edge("tools", "call_model")

    # Compile graph
    graph = builder.compile()

    # Run query
    result = await graph.ainvoke({
        "messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]
    })

    print("\nFinal Answer:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())