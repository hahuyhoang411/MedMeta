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
    synthesize_conclusion,
    answer_questions_with_llm,
    assess_target_text_suitability,
    evaluate_conclusion_match,
    generate_additional_questions,
    answer_additional_questions_with_llm
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
         # This will now effectively mean skipping retrieval and going to synthesis, which is fine.
         # If use_internal_knowledge was false, synthesis will report no docs.
         return []

    logging.info(f"Routing {len(valid_queries)} search queries to retrieval node.")
    send_actions = [
        Send("retrieve_documents", {"current_query": query})
        for query in valid_queries
    ]
    return send_actions

def decide_knowledge_path(state: MetaAnalysisState) -> str:
    """
    Decides the primary path for context generation or direct synthesis.
    Routes based on the 'synthesis_input_source' field in the state:
    - "llm_knowledge": Routes to answer questions using LLM's internal knowledge.
    - "retrieved_docs": Routes to generate search queries for document retrieval.
    - "target_text": Routes directly to synthesize conclusion using provided text.
    Note: "target_text_suitability" is handled by direct routing from START.
    """
    logging.info("--- Router: decide_knowledge_path ---")
    synthesis_source = state.get("synthesis_input_source")

    if synthesis_source == "llm_knowledge":
        logging.info("Path chosen: Use LLM internal knowledge (synthesis_input_source='llm_knowledge').")
        return "answer_questions_with_llm"
    elif synthesis_source == "target_text":
        logging.info("Path chosen: Use provided target reference text (synthesis_input_source='target_text'). Routing directly to synthesis.")
        return "synthesize_conclusion_direct" # New route name for clarity
    elif synthesis_source == "retrieved_docs":
        logging.info("Path chosen: Retrieve documents (synthesis_input_source='retrieved_docs').")
        return "generate_search_queries"
    else:
        # Default behavior if synthesis_input_source is not set or is an unexpected value.
        # For robustness, let's default to retrieval path.
        # The 'use_internal_knowledge' flag could be checked here as a fallback,
        # but it's better if 'synthesis_input_source' is always explicitly set by the caller.
        # For now, let's assume 'retrieved_docs' is the implicit default if 'synthesis_input_source' is missing.
        logging.warning(f"synthesis_input_source is '{synthesis_source}'. Defaulting to document retrieval path.")
        # To ensure state consistency if this path is taken by default:
        # state['synthesis_input_source'] = "retrieved_docs" # This modification should be done carefully or in entry point
        return "generate_search_queries"


def decide_evaluation_path(state: MetaAnalysisState) -> str:
    """
    Decides whether the conclusion is adequate or needs improvement through additional questions.
    Routes based on the evaluation score and whether we're already in the second iteration.
    """
    logging.info("--- Router: decide_evaluation_path ---")
    evaluation_score = state.get('conclusion_evaluation_score', 0)
    is_second_iteration = state.get('is_second_iteration', False)
    
    # If we're already in the second iteration, don't loop again to prevent infinite loops
    if is_second_iteration:
        logging.info("Second iteration completed. Ending process regardless of score.")
        return "end"
    
    # If score is 3 or higher (60%+), consider it adequate
    if evaluation_score >= 3:
        logging.info(f"Evaluation score {evaluation_score}/5 is adequate. Ending process.")
        return "end"
    else:
        logging.info(f"Evaluation score {evaluation_score}/5 is inadequate. Generating additional questions.")
        return "generate_additional_questions"

# --- Build the Graph ---

def build_graph(llm: BaseChatModel, retriever: Optional[ContextualCompressionRetriever]) -> Optional[StateGraph]:
    """
    Builds and compiles the LangGraph for the meta-analysis pipeline.

    Args:
        llm: The initialized language model instance.
        retriever: The initialized retriever instance.

    Returns:
        The compiled LangGraph application, or None if build fails.
    """
    if not llm: # Retriever is only needed for one path
        logging.error("LLM instance is missing. Cannot build graph.")
        return None

    try:
        graph_builder = StateGraph(MetaAnalysisState)

        # Add nodes, binding the LLM and retriever where needed
        graph_builder.add_node("generate_research_plan", lambda state: generate_research_plan(state, llm))
        graph_builder.add_node("generate_search_queries", lambda state: generate_search_queries(state, llm))
        # Retriever is only passed to retrieve_documents
        graph_builder.add_node("retrieve_documents", lambda state: retrieve_documents(state, retriever))
        graph_builder.add_node("answer_questions_with_llm", lambda state: answer_questions_with_llm(state, llm))
        graph_builder.add_node("assess_target_text_suitability", lambda state: assess_target_text_suitability(state, llm))
        graph_builder.add_node("synthesize_conclusion", lambda state: synthesize_conclusion(state, llm))
        
        # Enhanced llm_knowledge route nodes
        graph_builder.add_node("evaluate_conclusion_match", lambda state: evaluate_conclusion_match(state, llm))
        graph_builder.add_node("generate_additional_questions", lambda state: generate_additional_questions(state, llm))
        graph_builder.add_node("answer_additional_questions_with_llm", lambda state: answer_additional_questions_with_llm(state, llm))

        # Define edges
        # Conditional edge from START: route directly to assessment for target_text_suitability, otherwise to research plan
        graph_builder.add_conditional_edges(
            START,
            lambda state: "assess_target_text_suitability" if state.get("synthesis_input_source") == "target_text_suitability" else "generate_research_plan",
            {
                "assess_target_text_suitability": "assess_target_text_suitability",
                "generate_research_plan": "generate_research_plan"
            }
        )

        # Conditional edge: After research plan, decide which knowledge path to take
        graph_builder.add_conditional_edges(
            "generate_research_plan",
            decide_knowledge_path,
            {
                "generate_search_queries": "generate_search_queries",
                "answer_questions_with_llm": "answer_questions_with_llm",
                "synthesize_conclusion_direct": "synthesize_conclusion" # Added new route
            }
        )

        # Path 1: Document Retrieval
        graph_builder.add_conditional_edges(
            "generate_search_queries",
            route_to_retrieval,
            # If route_to_retrieval returns an empty list (no queries),
            # LangGraph needs a way to proceed. We want it to go to synthesis.
            # An empty list from route_to_retrieval means no Send actions are dispatched.
            # The graph will then look for a direct edge from 'generate_search_queries'
            # if no 'Send' actions are created or if all 'Send' paths complete.
            # To handle the "no valid queries" case from route_to_retrieval where it returns [],
            # and to ensure it correctly proceeds to synthesis, we can add a direct edge
            # for the case when no branches are taken.
            # However, LangGraph typically joins implicitly.
            # If route_to_retrieval returns [], it means no branches for 'retrieve_documents' are spawned.
            # The graph then needs to know where to go from 'generate_search_queries'.
            # The current setup of add_conditional_edges for route_to_retrieval
            # implicitly handles the "no branches" case by moving to the join point
            # for any 'Send' actions, or if none, it waits for an explicit next step from this node.
            # Let's ensure 'retrieve_documents' always leads to 'synthesize_conclusion'.
            # And if 'retrieve_documents' is skipped, 'generate_search_queries' should also lead to 'synthesize_conclusion'.

            # If 'route_to_retrieval' dispatches 'Send' actions to 'retrieve_documents', then 'retrieve_documents'
            # will eventually lead to 'synthesize_conclusion'.
            # If 'route_to_retrieval' returns [], meaning no queries to dispatch,
            # 'generate_search_queries' node effectively finishes. We need to connect it to 'synthesize_conclusion'.
             {"__END__": "synthesize_conclusion"} # If no Send actions, go to synthesize_conclusion
        )
        graph_builder.add_edge("retrieve_documents", "synthesize_conclusion")


        # Path 2: LLM Internal Knowledge (Enhanced with evaluation and feedback loop)
        graph_builder.add_edge("answer_questions_with_llm", "synthesize_conclusion")
        
        # Enhanced llm_knowledge route: evaluate conclusion after synthesis
        graph_builder.add_conditional_edges(
            "synthesize_conclusion",
            lambda state: "evaluate_conclusion_match" if state.get("synthesis_input_source") == "llm_knowledge" else "end",
            {
                "evaluate_conclusion_match": "evaluate_conclusion_match",
                "end": END
            }
        )
        
        # Route based on evaluation score and iteration status
        graph_builder.add_conditional_edges(
            "evaluate_conclusion_match",
            decide_evaluation_path,
            {
                "generate_additional_questions": "generate_additional_questions",
                "end": END
            }
        )
        
        # Feedback loop for additional questions
        graph_builder.add_edge("generate_additional_questions", "answer_additional_questions_with_llm")
        graph_builder.add_edge("answer_additional_questions_with_llm", "synthesize_conclusion")

        # Path 3: Target Text Suitability Assessment
        graph_builder.add_edge("assess_target_text_suitability", END)

        # Compile the graph
        logging.info("Compiling the LangGraph...")
        app = graph_builder.compile()
        logging.info("LangGraph compiled successfully.")
        return app

    except Exception as e:
        logging.error(f"Failed to build or compile LangGraph: {e}", exc_info=True)
        return None
