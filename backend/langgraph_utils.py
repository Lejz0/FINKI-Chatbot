from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config import settings

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class ChatAgent:
    def __init__(self, model, tools, graph=None, checkpointer=None, system_prompt=""):
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        if graph and checkpointer:
            self.graph = graph.compile(checkpointer=checkpointer)
        else:
            self.graph = None

    @classmethod
    async def create(cls, model, tools, system_prompt="You're a helpful assistant that answer's the user question. The questions the user asks are in Macedonian, so is the data"):
        agent = cls(model, tools, system_prompt=system_prompt)
        
        graph = StateGraph(AgentState)

        graph.add_node("initial_llm", agent.call_llm) 
        graph.add_node("final_llm", agent.call_llm, metadata={"tags": ["final_node"]})
        graph.add_node("function", agent.execute_function)

        graph.add_conditional_edges(
            "initial_llm",
            agent.exists_function_calling,
            {True: "function", False: END}
        )

        graph.add_edge("function", "final_llm")
        graph.add_edge("final_llm", END)
        graph.set_entry_point("initial_llm")

        connection_pool = await agent.get_connection_pool()
        checkpointer = AsyncPostgresSaver(connection_pool)
        await checkpointer.setup()

        agent.graph = graph.compile(checkpointer=checkpointer)
        return agent

    def call_llm(self, state: AgentState):
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def execute_function(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for tool in tool_calls:
            if tool['name'] not in self.tools:
                result = "Error: There's no such tool, please try again"
            else:
                result = self.tools[tool['name']].invoke(tool['args'])

            results.append(ToolMessage(
                tool_call_id=tool['id'],
                name=tool['name'],
                content=str(result)
            ))

        return {'messages': results}

    @staticmethod
    async def get_connection_pool():
        pool = AsyncConnectionPool(
            settings.postgres_url,
            open=False,
            kwargs={
                "autocommit": True,
                "connect_timeout": 5,
                "prepare_threshold": None,
            }
        )
        await pool.open()
        return pool

    def exists_function_calling(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    async def get_streaming_response(self, user_input: str, thread_id: str):
        state = {"messages": [HumanMessage(content=user_input)]}
        async for message, metadata in self.graph.astream(
            state,
            stream_mode="messages",
            config={"configurable": {"thread_id": thread_id}},
        ):
            if metadata.get("tags") and "final_node" in metadata["tags"]:
                yield message.content