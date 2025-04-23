import logging
from typing import List, Dict, Any, Optional

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.retrievers import ContextualCompressionRetriever

# Relative imports for state, models, and nodes within the same package
from .state_models import MetaAnalysisState
from .nodes import (
    generate_research_plan,
    generate_search_queries,
    retrieve_documents,
    synthesize_conclusion
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def route_to_retrieval(state: MetaAnalysisState) -> List[Send]:
    """
    Routes to the retrieval node for each valid generated search query.
    Uses langgraph.types.Send to dispatch parallel calls.
    """
    logging.info("--- Router: route_to_retrieval ---")
    queries = state.get('search_queries', []) # Get queries generated in the previous step
    valid_queries = [q for q in queries if q and isinstance(q, str) and q.strip()]

    if not valid_queries:
         logging.warning("No valid search queries generated or found in state. Routing to synthesis.")
         # If no queries, the list of Send actions is empty, graph proceeds to the join point (synthesis)
         return [] # Empty list signifies moving to the node that waits for all branches (synthesize)

    logging.info(f"Routing {len(valid_queries)} search queries to retrieval node.")
    # Create a Send action for each valid query.
    # The dictionary passed is the *update* to the state for that specific branch.
    # We set 'current_query' for the retrieve_documents node to use.
    send_actions = [
        Send("retrieve_documents", {"current_query": query})
        for query in valid_queries
    ]
    return send_actions


# --- Build the Graph ---

def build_graph(llm: BaseChatModel, retriever: ContextualCompressionRetriever) -> Optional[StateGraph]:
    """
    Builds and compiles the LangGraph for the meta-analysis pipeline.

    Args:
        llm: The initialized language model instance.
        retriever: The initialized retriever instance.

    Returns:
        The compiled LangGraph application, or None if build fails.
    """
    if not llm or not retriever:
        logging.error("LLM or Retriever instance is missing. Cannot build graph.")
        return None

    try:
        graph_builder = StateGraph(MetaAnalysisState)

        # Add nodes, binding the LLM and retriever where needed
        graph_builder.add_node("generate_research_plan", lambda state: generate_research_plan(state, llm))
        graph_builder.add_node("generate_search_queries", lambda state: generate_search_queries(state, llm))
        graph_builder.add_node("retrieve_documents", lambda state: retrieve_documents(state, retriever))
        graph_builder.add_node("synthesize_conclusion", lambda state: synthesize_conclusion(state, llm))

        # Define edges
        graph_builder.add_edge(START, "generate_research_plan")
        graph_builder.add_edge("generate_research_plan", "generate_search_queries")

        # Conditional edge: After query generation, fan out to retrieve for each query
        graph_builder.add_conditional_edges(
            "generate_search_queries",
            route_to_retrieval,
            # The key "" indicates that the next step depends *only* on the routing logic result
            # If route_to_retrieval returns [], it implicitly goes to the node that joins the branches.
            # If it returns Send actions, those branches execute.
        )

        # After all retrieval branches complete (implicitly handled by LangGraph),
        # proceed to synthesis. The 'retrieve_documents' node outputs to 'retrieved_docs',
        # which are automatically aggregated by the `operator.add` annotation in the state.
        graph_builder.add_edge("retrieve_documents", "synthesize_conclusion")

        # Final step
        graph_builder.add_edge("synthesize_conclusion", END)

        # Compile the graph
        logging.info("Compiling the LangGraph...")
        app = graph_builder.compile()
        logging.info("LangGraph compiled successfully.")
        return app

    except Exception as e:
        logging.error(f"Failed to build or compile LangGraph: {e}", exc_info=True)
        return None