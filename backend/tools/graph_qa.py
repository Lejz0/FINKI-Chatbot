from pydantic import BaseModel, Field
from langchain_neo4j import GraphCypherQAChain
from langchain.tools import tool
from typing import Any
from functools import lru_cache
from neo4j_utils import get_graph
from langchain_utils import get_llm

class GraphQAInput(BaseModel):
    query: str = Field(description="The full natural language user question. Do not simplify, shorten, or rephrase the query.")

def create_graph_qa_tool(llm: Any, graph: Any):
    """Create a GraphQA tool with the specified LLM and graph."""
    
    @tool(args_schema=GraphQAInput)
    def graph_qa_tool(query: str):
        """Query a graph database with natural language and get relevant information."""

        chain = GraphCypherQAChain.from_llm(
            llm,
            graph=graph, 
            verbose=True, 
            return_direct = True,
            allow_dangerous_requests=True
        )
        
        result = chain.run(query)
        return result
    
    return graph_qa_tool

@lru_cache(maxsize=1)
def get_graph_qa_tool() -> Any:
    llm = get_llm()
    graph = get_graph()
    return create_graph_qa_tool(llm, graph)