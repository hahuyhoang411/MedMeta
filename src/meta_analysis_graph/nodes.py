import logging
from typing import Dict, List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .state_models import MetaAnalysisState, ResearchPlan, SearchQueries, LLMAnsweredQuestion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder for structured LLM answers - consider moving to state_models.py
# class LLMAnsweredQuestion(TypedDict):
#     question: str # The original key research question
#     comprehensive_answer: str # LLM's detailed answer, potentially including sub-questions and their answers

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


def answer_questions_with_llm(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Uses the LLM's internal knowledge to answer the key research questions.
    The LLM is prompted to also generate and answer sub-questions if relevant.
    """
    logging.info("--- Node: answer_questions_with_llm ---")
    plan = state.get('research_plan')

    if not plan or not isinstance(plan, ResearchPlan) or not plan.key_questions:
        logging.error("Research plan or key questions missing. Cannot answer questions with LLM.")
        return {"llm_generated_answers": []}

    answered_questions: List[LLMAnsweredQuestion] = []

    for question in plan.key_questions:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert researcher with broad knowledge. For the given research question, provide a comprehensive answer based on your internal knowledge. If applicable, identify 2-3 critical sub-questions that arise from this main question and provide detailed answers to those as well within your response. Structure your entire response as a single coherent text."),
            ("human", "Research Question: {question}")
        ])
        try:
            answer_chain = prompt | llm
            response = answer_chain.invoke({"question": question})
            answered_questions.append({
                "question": question,
                "comprehensive_answer": response.content
            })
            logging.info(f"LLM answered question: '{question}'")
        except Exception as e:
            logging.error(f"Error answering question '{question}' with LLM: {e}", exc_info=True)
            answered_questions.append({
                "question": question,
                "comprehensive_answer": f"Error generating answer for this question: {e}"
            })

    return {"llm_generated_answers": answered_questions}


def assess_target_text_suitability(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Assesses whether the provided target reference text abstracts contain sufficient
    information to recreate the original conclusion from the MedMeta dataset.
    Returns a suitability score from 0-5 and detailed assessment.
    """
    logging.info("--- Node: assess_target_text_suitability ---")
    topic = state['research_topic']
    target_text = state.get('target_reference_text')
    original_conclusion = state.get('original_conclusion')

    if not target_text:
        logging.warning("No target reference text found for suitability assessment.")
        return {
            "suitability_score": 0,
            "suitability_assessment": "No target reference text was provided for assessment."
        }

    if not original_conclusion:
        logging.warning("No original conclusion found for suitability assessment.")
        return {
            "suitability_score": 0,
            "suitability_assessment": "No original conclusion was provided for assessment."
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert research analyst tasked with evaluating whether provided abstracts "
         "contain sufficient information to recreate a specific original conclusion from a meta-analysis. "
         "Your task is to assess if someone could reasonably arrive at the same conclusion as the "
         "original authors by reading only the provided abstracts.\n\n"
         "Provide your assessment as:\n"
         "1. A detailed evaluation including:\n"
         "   - What key information from the original conclusion is present in the abstracts\n"
         "   - What important information from the original conclusion might be missing\n"
         "   - Whether the abstracts provide sufficient evidence to support the original conclusion\n"
         "   - Any gaps or limitations that would prevent recreating the original conclusion\n\n"
         "2. A score from 0-5 where:\n"
         "   - 0 = Completely insufficient - Cannot recreate conclusion (0% confidence)\n"
         "   - 1 = Very insufficient - Major gaps prevent conclusion recreation (20% confidence)\n"
         "   - 2 = Insufficient - Significant missing information (40% confidence)\n"
         "   - 3 = Moderately sufficient - Some gaps but core elements present (60% confidence)\n"
         "   - 4 = Good sufficiency - Minor gaps but conclusion is supportable (80% confidence)\n"
         "   - 5 = Excellent sufficiency - All needed information present to recreate conclusion (100% confidence)\n\n"
         "Focus specifically on whether the abstracts support the original conclusion's claims, "
         "findings, and recommendations."
        ),
        ("human", 
         "Research Topic: {topic}\n\n"
         "Original Conclusion (to be recreated):\n{original_conclusion}\n\n"
         "Target Reference Text Abstracts:\n{target_text}\n\n"
         "Please assess whether these abstracts contain sufficient information to recreate "
         "the original conclusion above. Provide both a detailed evaluation and numerical score (0-5)."
        )
    ])

    try:
        assessment_chain = prompt | llm
        response = assessment_chain.invoke({
            "topic": topic,
            "original_conclusion": original_conclusion,
            "target_text": target_text
        })
        
        assessment_text = response.content
        
        # Extract score from the response with improved parsing
        import re
        extracted_score = None
        
        # Method 1: Look for "Score:" or "Rating:" followed by a number (handles whitespace and newlines)
        score_patterns = [
            r'(?:score|rating)\s*:?\s*\n*\s*([0-5])',  # Score: 3 or Score:\n\n3
            r'(?:score|rating)\s*:?\s*\n*\s*([0-5])\s*/\s*5',  # Score: 3/5 or Score:\n3/5
            r'(?:score|rating)\s*:?\s*\n*\s*([0-5])\s*out\s*of\s*5',  # Score: 3 out of 5
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, assessment_text.lower())
            if match:
                try:
                    extracted_score = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Method 2: Look for standalone numbers with score context
        if extracted_score is None:
            # Look for patterns like "## Score:\n\n2" or "The score is 3"
            context_patterns = [
                r'(?:score|rating|assign|give|rate)\s*(?:is|of)?\s*:?\s*\n*\s*([0-5])',
                r'([0-5])\s*/\s*5',  # X/5 pattern
                r'([0-5])\s*out\s*of\s*5',  # X out of 5 pattern
            ]
            
            for pattern in context_patterns:
                matches = re.findall(pattern, assessment_text.lower())
                if matches:
                    try:
                        # Take the last occurrence (often the final score)
                        extracted_score = int(matches[-1])
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Method 3: Look at the last few lines for a standalone number (common LLM pattern)
        if extracted_score is None:
            lines = assessment_text.strip().split('\n')
            for line in reversed(lines[-5:]):  # Check last 5 lines
                line = line.strip()
                if line and re.match(r'^[0-5]$', line):
                    try:
                        extracted_score = int(line)
                        break
                    except ValueError:
                        continue
        
        # Method 4: Fallback - find any single digit 0-5 in the text
        if extracted_score is None:
            all_digits = re.findall(r'\b([0-5])\b', assessment_text)
            if all_digits:
                try:
                    # Take the last occurrence as it's often the final score
                    extracted_score = int(all_digits[-1])
                except (ValueError, IndexError):
                    pass
        
        # Ensure score is within valid range, default to 0 if still None
        extracted_score = max(0, min(5, extracted_score)) if extracted_score is not None else 0
        
        logging.info(f"Suitability assessment completed. Score: {extracted_score}/5")
        logging.info(f"Assessment preview: {assessment_text[:200]}...")
        
        return {
            "suitability_score": extracted_score,
            "suitability_assessment": assessment_text
        }
        
    except Exception as e:
        logging.error(f"Error during suitability assessment: {e}", exc_info=True)
        return {
            "suitability_score": 0,
            "suitability_assessment": f"Error occurred during assessment: {e}"
        }


def synthesize_conclusion(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Synthesizes a final conclusion.
    The source of information for synthesis is determined by 'synthesis_input_source' in the state:
    - "retrieved_docs": Uses aggregated retrieved documents.
    - "llm_knowledge": Uses LLM-generated answers.
    - "target_text": Uses a directly provided target reference text.
    """
    logging.info("--- Node: synthesize_conclusion ---")
    topic = state['research_topic']
    plan = state.get('research_plan')
    synthesis_source = state.get('synthesis_input_source', "retrieved_docs") # Default to retrieved_docs if not specified

    key_questions_str = "N/A"
    if plan and isinstance(plan, ResearchPlan) and plan.key_questions:
        key_questions_str = "\n".join(f"- {q}" for q in plan.key_questions)
    else:
        logging.warning("Research plan or key questions missing for synthesis context. Proceeding without them.")

    context_string = ""
    synthesis_input_type = ""

    if synthesis_source == "llm_knowledge":
        logging.info("Synthesizing conclusion from LLM-generated answers.")
        llm_answers = state.get('llm_generated_answers', [])
        if not llm_answers:
            logging.warning("No LLM-generated answers found to synthesize a conclusion (source: llm_knowledge).")
            return {"final_conclusion": "No LLM-generated answers were available to synthesize a conclusion."}

        context_string = "\n\n---\n\n".join(
            [f"Question: {ans['question']}\nAnswer:\n{ans['comprehensive_answer']}"
             for ans in llm_answers]
        )
        synthesis_input_type = "LLM-Generated Answers to Research Questions"
    elif synthesis_source == "target_text":
        logging.info("Synthesizing conclusion from provided target reference text.")
        target_text = state.get('target_reference_text')
        if not target_text:
            logging.warning("No target reference text found to synthesize a conclusion (source: target_text).")
            return {"final_conclusion": "No target reference text was provided to synthesize a conclusion."}
        context_string = target_text
        synthesis_input_type = "Provided Target Reference Text"
    elif synthesis_source == "retrieved_docs": # Default or explicitly set
        logging.info("Synthesizing conclusion from retrieved documents.")
        retrieved_docs = state.get('retrieved_docs', [])
        seen_content = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        logging.info(f"Aggregated {len(retrieved_docs)} docs, reduced to {len(unique_docs)} unique docs for synthesis (source: retrieved_docs).")

        if not unique_docs:
            logging.warning("No unique documents found to synthesize a conclusion (source: retrieved_docs).")
            return {"final_conclusion": "No relevant documents were found or retrieved to synthesize a conclusion."}

        context_string = "\n\n---\n\n".join(
            [f"Source PMID: {doc.metadata.get('PMID', 'N/A')}\nYear: {doc.metadata.get('Year', 'N/A')}\nContent:\n{doc.page_content}"
             for doc in unique_docs[:50]] # Limit context to first 50 unique docs
        )
        synthesis_input_type = "Retrieved Study Context"
    else:
        logging.error(f"Invalid synthesis_input_source: {synthesis_source}. Defaulting to no conclusion.")
        return {"final_conclusion": "Invalid input source specified for synthesis."}

    prompt_template_str = (
        "You are a research analyst tasked with creating the final concluding summary for a meta-analysis or systematic review. "
        "Based *only* on the provided context, synthesize a concise wrap-up conclusion. "
        "This conclusion should address the key research questions (if available), identify major themes, consistent findings, discrepancies, or gaps mentioned. "
        "Frame your response as a final summary of the evidence or information presented in the context. Do not add external knowledge."
        "\n\nResearch Topic: {topic}"
        "\n\nResearch Plan Key Questions:\n{key_questions}"
        f"\n\n{synthesis_input_type}:\n{{context}}"
        "\n\nSynthesize a concise conclusion based *only* on the provided context:"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template_str.split("\n\nResearch Topic:")[0].strip()), # System part
        ("human", "\nResearch Topic: {topic}" + prompt_template_str.split("\nResearch Topic: {topic}", 1)[1]) # Human part
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