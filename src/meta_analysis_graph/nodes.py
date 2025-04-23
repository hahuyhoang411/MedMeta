import logging
from typing import Dict, List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .state_models import MetaAnalysisState, ResearchPlan, SearchQueries # Relative import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_research_plan(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Generates a structured research plan based on the initial topic using the LLM.
    """
    logging.info("--- Node: generate_research_plan ---")
    topic = state['research_topic']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant skilled in formulating structured research plans for systematic reviews or meta-analyses. Given a research topic, create a concise plan including background context, 5 key research questions, and a brief summary of the search strategy/concepts."),
        ("human", "Generate a research plan for the topic: {topic}")
    ])

    try:
        plan_chain = prompt | llm.with_structured_output(ResearchPlan)
        response = plan_chain.invoke({"topic": topic})

        logging.info(f"Generated Research Plan:\nBackground: {response.background}\nKey Questions: {response.key_questions}\nStrategy Summary: {response.search_strategy_summary}")
        return {"research_plan": response}
    except Exception as e:
        logging.error(f"Error in generate_research_plan: {e}", exc_info=True)
        # Return a default or raise error depending on desired graph behavior
        return {"research_plan": ResearchPlan(background="Error generating plan", key_questions=[], search_strategy_summary="")}


def generate_search_queries(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Generates optimized search queries based on the research plan using the LLM.
    """
    logging.info("--- Node: generate_search_queries ---")
    plan = state.get('research_plan')
    if not plan or not isinstance(plan, ResearchPlan) or not plan.key_questions:
         logging.error("Research plan is missing or invalid in state. Cannot generate queries.")
         # Returning empty list allows graph to proceed to synthesis, which handles no docs
         return {"search_queries": []}

    # Create a context string from the plan for the prompt
    plan_context = f"Background: {plan.background}\nKey Questions:\n" + "\n".join(f"- {q}" for q in plan.key_questions) + f"\nSearch Strategy Summary: {plan.search_strategy_summary}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in creating search queries for biomedical literature databases like PubMed, optimized for modern retrieval systems (BM25 keyword search and dense vector search). Based on the provided research plan, generate a list of 5 effective search queries. Focus on extracting key concepts, relevant keywords, and potential synonyms. **Do NOT use boolean operators (AND, OR, NOT)** as the retrieval system handles term weighting. Aim for queries that capture the core topics for both keyword relevance and semantic similarity."),
        ("human", "Generate 5 search queries based on this research plan:\n\n{plan_context}")
    ])

    try:
        query_chain = prompt | llm.with_structured_output(SearchQueries)
        response = query_chain.invoke({"plan_context": plan_context})
        queries = response.queries if response else []
        logging.info(f"Generated Search Queries: {queries}")
        # Filter out empty strings just in case
        valid_queries = [q for q in queries if q and q.strip()]
        return {"search_queries": valid_queries}
    except Exception as e:
        logging.error(f"Error in generate_search_queries: {e}", exc_info=True)
        return {"search_queries": []}


def retrieve_documents(state: MetaAnalysisState, retriever: ContextualCompressionRetriever) -> Dict[str, Any]:
    """
    Retrieves documents for the 'current_query' passed into this graph branch.
    Returns documents under the key 'retrieved_docs' for aggregation.
    """
    query_to_retrieve = state['current_query'] # This key is set by the Send action
    logging.info(f"--- Node: retrieve_documents (Query: '{query_to_retrieve}') ---")
    if not query_to_retrieve or not isinstance(query_to_retrieve, str):
        logging.warning("Invalid or empty query received for retrieval. Skipping.")
        return {"retrieved_docs": []} # Return empty list for aggregation

    if retriever is None:
        logging.error("Retriever not provided to retrieve_documents node.")
        return {"retrieved_docs": []}

    try:
        docs: List[Document] = retriever.invoke(query_to_retrieve)
        logging.info(f"Retrieved {len(docs)} documents for query '{query_to_retrieve}'.")
        # The key here MUST match the key aggregated in the state (retrieved_docs)
        return {"retrieved_docs": docs}
    except Exception as e:
        logging.error(f"Error during document retrieval for query '{query_to_retrieve}': {e}", exc_info=True)
        return {"retrieved_docs": []} # Return empty list on error


def synthesize_conclusion(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Synthesizes a final conclusion from all aggregated retrieved documents.
    """
    logging.info("--- Node: synthesize_conclusion ---")
    topic = state['research_topic']
    plan = state.get('research_plan')
    retrieved_docs = state.get('retrieved_docs', []) # Aggregated docs

    # De-duplicate documents based on page content (simple approach)
    seen_content = set()
    unique_docs = []
    for doc in retrieved_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)

    logging.info(f"Aggregated {len(retrieved_docs)} docs, reduced to {len(unique_docs)} unique docs for synthesis.")

    if not unique_docs:
        logging.warning("No unique documents found to synthesize a conclusion.")
        return {"final_conclusion": "No relevant documents were found or retrieved to synthesize a conclusion."}

    if not plan or not isinstance(plan, ResearchPlan) or not plan.key_questions:
        logging.warning("Research plan or key questions missing for synthesis context. Proceeding without them.")
        key_questions_str = "N/A"
    else:
        key_questions_str = "\n".join(f"- {q}" for q in plan.key_questions)


    # Limit context size if necessary, although Gemini Flash has large context
    # Consider summarizing long docs or selecting top passages if context exceeds limits
    context_string = "\n\n---\n\n".join(
        [f"Source PMID: {doc.metadata.get('PMID', 'N/A')}\nYear: {doc.metadata.get('Year', 'N/A')}\nContent:\n{doc.page_content}"
         for doc in unique_docs[:50]] # Limit context to first 50 unique docs for safety
        )


    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research analyst summarizing findings for a meta-analysis or systematic review. Based *only* on the provided context from retrieved studies, synthesize a concise conclusion addressing the key research questions outlined in the plan (if available). Identify major themes, consistent findings, discrepancies, or gaps mentioned across the studies. Do not add external knowledge. Focus on summarizing the evidence presented."),
        ("human", "Research Topic: {topic}\n\nResearch Plan Key Questions:\n{key_questions}\n\nRetrieved Study Context:\n{context}\n\nSynthesize a concise conclusion based *only* on the provided context:")
    ])

    try:
        synthesis_chain = prompt | llm
        response = synthesis_chain.invoke({
            "topic": topic,
            "key_questions": key_questions_str,
            "context": context_string
        })
        conclusion = response.content
        logging.info(f"Generated Conclusion: {conclusion[:500]}...") # Log start of conclusion
        return {"final_conclusion": conclusion}
    except Exception as e:
        logging.error(f"Error during conclusion synthesis: {e}", exc_info=True)
        return {"final_conclusion": "Error occurred during synthesis."}