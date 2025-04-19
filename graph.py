from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agent import perform_rag, detect_domains, domain_router, generate_answer, confidence_filter, output_node

class AppState(TypedDict):
    query: str
    context: str
    domains: List[str]
    responses: List[tuple]
    final_response: str

graph = StateGraph(AppState)

graph.add_node("perform_rag", perform_rag)
graph.add_node("detect_domains", detect_domains)
graph.add_node("domain_router", domain_router)
graph.add_node("generate_answer", generate_answer)
graph.add_node("confidence_filter", confidence_filter)
graph.add_node("output_node", output_node)

graph.set_entry_point("perform_rag")
graph.add_edge("perform_rag", "detect_domains")
graph.add_edge("detect_domains", "domain_router")
graph.add_edge("domain_router", "generate_answer")
graph.add_edge("generate_answer", "confidence_filter")
graph.add_edge("confidence_filter", "output_node")
graph.add_edge("output_node", END)

app = graph.compile()
