from langchain_neo4j import Neo4jGraph
from functools import lru_cache

from config import settings

@lru_cache()
def get_graph():
    graph = Neo4jGraph(url=settings.database_url, 
                       username=settings.database_username, 
                       password=settings.database_password, 
                       enhanced_schema=True)
    
    return graph