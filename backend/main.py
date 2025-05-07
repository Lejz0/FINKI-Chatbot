from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic_models import QueryInput
from neo4j_utils import get_graph
from langchain_utils import get_llm
from langgraph_utils import ChatAgent
from tools.graph_qa import get_graph_qa_tool

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generator(query: str, session_id: str, llm):
    tool = get_graph_qa_tool()
    agent = await ChatAgent.create(llm, [tool])

    async for message in agent.get_streaming_response(user_input=query, thread_id=session_id):
        yield f"data: {message}\n\n"

@app.post("/chat")
async def root(query: QueryInput, graph = Depends(get_graph), llm = Depends(get_llm)):
    return StreamingResponse(generator(query=query.question, session_id=query.session_id, llm=llm, graph=graph), media_type="text/event_stream")

# chat_sessions= {}

# async def generate_chat_message(query: str, session_id : str, llm, graph):
#     chain = GraphCypherQAChain.from_llm(llm, 
#                                         graph=graph, 
#                                         verbose=True, 
#                                         allow_dangerous_requests=True, 
#                                         return_intermediate_steps=True,)

    
#     if session_id in chat_sessions:
#         history = chat_sessions[session_id]
#     else:
#         session_id = str(uuid.uuid4())
#         chat_sessions[session_id] = []
#         history = []
    
#     print(f" History: {history}")
    
#     history.append(HumanMessage(content=query))

#     result = await chain.ainvoke(history)

#     context = result.get("intermediate_steps")

#     inputs = {"question": query, "context": context}
    
#     full_response = ""

#     async for chunk in chain.qa_chain.astream(inputs):
#         full_response += chunk
#         yield f"data: {chunk}\n\n"

#     history.append(AIMessage(content=full_response))

#     chat_sessions[session_id] = history
#     yield f"data: [SESSION_ID:{session_id}]\n\n"




# @app.post("/chat", response_model=QueryResponse)
# def get_chat(query: QueryInput, graph = Depends(get_graph), llm = Depends(get_llm)):

#     chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True, allow_dangerous_requests=True)

#     answer = chain.invoke({"query": query.question})['result']

#     return QueryResponse(answer=answer)


