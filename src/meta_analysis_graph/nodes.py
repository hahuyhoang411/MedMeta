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
        additional_answers = state.get('additional_answers', [])
        
        # Combine original and additional answers if available
        all_answers = llm_answers + (additional_answers or [])
        
        if not all_answers:
            logging.warning("No LLM-generated answers found to synthesize a conclusion (source: llm_knowledge).")
            return {"final_conclusion": "No LLM-generated answers were available to synthesize a conclusion."}

        context_string = "\n\n---\n\n".join(
            [f"Question: {ans['question']}\nAnswer:\n{ans['comprehensive_answer']}"
             for ans in all_answers]
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
        "You are a research analyst tasked with drafting the **primary concluding statement** "
        "for a meta-analysis or systematic review. Your goal is to distill the provided context "
        "into the **single most important and specific takeaway message**, as if you were presenting "
        "the main result of the study.\n\n"
        "Based *strictly* on the provided context:\n"
        "1. Identify the **central, affirmative findings** or **key definitive statements** made. "
        "What is the most crucial outcome, comparison, or result reported?\n"
        "2. Capture any **critical quantifications, effect sizes, or specific comparisons** "
        "that are central to this main finding.\n"
        "3. Include any **essential caveats, limitations, or conditions** that are "
        "directly tied to and qualify this primary finding.\n"
        "4. The conclusion should be **highly focused and concise**, reflecting the punchline "
        "of the research. Avoid general summaries of the entire field or background "
        "information from the context.\n"
        "5. Do not introduce external knowledge or comment on the completeness of the provided context."
        "\n\nResearch Topic: {topic}"
        f"\n\n{synthesis_input_type}:\n{{context}}"
        "\n\nSynthesize the primary concluding statement based *only* on the provided context, focusing on the most direct and impactful findings:"
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


def evaluate_conclusion_match(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Evaluates whether the generated conclusion adequately matches and addresses the research topic.
    Returns a score from 0-5 and detailed feedback.
    """
    logging.info("--- Node: evaluate_conclusion_match ---")
    topic = state['research_topic']
    conclusion = state.get('final_conclusion')
    plan = state.get('research_plan')
    
    if not conclusion:
        logging.warning("No conclusion found for evaluation.")
        return {
            "conclusion_evaluation_score": 0,
            "conclusion_evaluation_feedback": "No conclusion was provided for evaluation."
        }
    
    # Include research plan context if available
    plan_context = ""
    if plan and isinstance(plan, ResearchPlan):
        plan_context = f"\nResearch Plan Background: {plan.background}\nKey Questions: {', '.join(plan.key_questions)}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert research evaluator tasked with assessing whether a generated conclusion "
         "adequately addresses and matches the given research topic. Your evaluation should consider:\n\n"
         "1. **Topic Relevance**: Does the conclusion directly address the research topic?\n"
         "2. **Comprehensiveness**: Does the conclusion cover the key aspects expected for this topic?\n"
         "3. **Specificity**: Is the conclusion specific enough to be meaningful for the research topic?\n"
         "4. **Coherence**: Is the conclusion logically coherent with respect to the topic?\n"
         "5. **Completeness**: Does the conclusion feel complete for addressing the research topic?\n\n"
         "Provide your assessment as:\n"
         "1. A detailed evaluation explaining what works well and what might be missing or inadequate\n"
         "2. A score from 0-5 where:\n"
         "   - 0 = Completely inadequate - Does not address the topic (0% match)\n"
         "   - 1 = Very inadequate - Major gaps in addressing the topic (20% match)\n"
         "   - 2 = Inadequate - Significant missing aspects (40% match)\n"
         "   - 3 = Moderately adequate - Some gaps but core topic addressed (60% match)\n"
         "   - 4 = Good - Minor gaps but well addresses the topic (80% match)\n"
         "   - 5 = Excellent - Comprehensively and specifically addresses the topic (100% match)\n\n"
         "Focus on whether the conclusion is sufficient for someone researching this specific topic."
        ),
        ("human", 
         "Research Topic: {topic}{plan_context}\n\n"
         "Generated Conclusion:\n{conclusion}\n\n"
         "Please evaluate whether this conclusion adequately matches and addresses the research topic. "
         "Provide both a detailed evaluation and numerical score (0-5)."
        )
    ])
    
    try:
        evaluation_chain = prompt | llm
        response = evaluation_chain.invoke({
            "topic": topic,
            "plan_context": plan_context,
            "conclusion": conclusion
        })
        
        evaluation_text = response.content
        
        # Extract score from the response (similar to assess_target_text_suitability)
        import re
        extracted_score = None
        
        # Method 1: Look for "Score:" or "Rating:" followed by a number
        score_patterns = [
            r'(?:score|rating)\s*:?\s*\n*\s*([0-5])',
            r'(?:score|rating)\s*:?\s*\n*\s*([0-5])\s*/\s*5',
            r'(?:score|rating)\s*:?\s*\n*\s*([0-5])\s*out\s*of\s*5',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, evaluation_text.lower())
            if match:
                try:
                    extracted_score = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Method 2: Look for standalone numbers with score context
        if extracted_score is None:
            context_patterns = [
                r'(?:score|rating|assign|give|rate)\s*(?:is|of)?\s*:?\s*\n*\s*([0-5])',
                r'([0-5])\s*/\s*5',
                r'([0-5])\s*out\s*of\s*5',
            ]
            
            for pattern in context_patterns:
                matches = re.findall(pattern, evaluation_text.lower())
                if matches:
                    try:
                        extracted_score = int(matches[-1])
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Method 3: Fallback - look at last few lines for standalone number
        if extracted_score is None:
            lines = evaluation_text.strip().split('\n')
            for line in reversed(lines[-5:]):
                line = line.strip()
                if line and re.match(r'^[0-5]$', line):
                    try:
                        extracted_score = int(line)
                        break
                    except ValueError:
                        continue
        
        # Ensure score is within valid range, default to 0 if still None
        extracted_score = max(0, min(5, extracted_score)) if extracted_score is not None else 0
        
        logging.info(f"Conclusion evaluation completed. Score: {extracted_score}/5")
        logging.info(f"Evaluation preview: {evaluation_text[:200]}...")
        
        return {
            "conclusion_evaluation_score": extracted_score,
            "conclusion_evaluation_feedback": evaluation_text
        }
        
    except Exception as e:
        logging.error(f"Error during conclusion evaluation: {e}", exc_info=True)
        return {
            "conclusion_evaluation_score": 0,
            "conclusion_evaluation_feedback": f"Error occurred during evaluation: {e}"
        }


def generate_additional_questions(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Generates 5 additional sub-questions that are different from the original ones
    to improve the conclusion when evaluation shows inadequate topic matching.
    """
    logging.info("--- Node: generate_additional_questions ---")
    topic = state['research_topic']
    plan = state.get('research_plan')
    original_questions = []
    
    if plan and isinstance(plan, ResearchPlan):
        original_questions = plan.key_questions
    
    # Get evaluation feedback to understand what was missing
    evaluation_feedback = state.get('conclusion_evaluation_feedback', '')
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a research assistant expert at formulating targeted research questions. "
         "Given a research topic, original questions that were already asked, and feedback about "
         "what was missing from the initial conclusion, generate 5 NEW and DIFFERENT sub-questions "
         "that will help address the gaps and improve understanding of the research topic.\n\n"
         "Your new questions should:\n"
         "1. Be completely different from the original questions\n"
         "2. Address specific gaps mentioned in the evaluation feedback\n"
         "3. Explore different angles, perspectives, or aspects of the topic\n"
         "4. Be specific and actionable for research purposes\n"
         "5. Help fill in missing information to better address the research topic"
        ),
        ("human", 
         "Research Topic: {topic}\n\n"
         "Original Questions Already Asked:\n{original_questions}\n\n"
         "Evaluation Feedback (what was missing/inadequate):\n{evaluation_feedback}\n\n"
         "Generate 5 NEW sub-questions that are different from the original ones and will help "
         "address the gaps identified in the evaluation feedback:"
        )
    ])
    
    try:
        original_questions_str = "\n".join(f"- {q}" for q in original_questions) if original_questions else "None"
        
        question_chain = prompt | llm.with_structured_output(SearchQueries)
        response = question_chain.invoke({
            "topic": topic,
            "original_questions": original_questions_str,
            "evaluation_feedback": evaluation_feedback
        })
        
        additional_questions = response.queries if response else []
        
        # Filter out empty strings and ensure we have valid questions
        valid_questions = [q for q in additional_questions if q and q.strip()]
        
        logging.info(f"Generated {len(valid_questions)} additional questions: {valid_questions}")
        
        return {
            "additional_questions": valid_questions,
            "is_second_iteration": True  # Mark that we're in the second iteration
        }
        
    except Exception as e:
        logging.error(f"Error generating additional questions: {e}", exc_info=True)
        return {
            "additional_questions": [],
            "is_second_iteration": True
        }


def answer_additional_questions_with_llm(state: MetaAnalysisState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Uses the LLM's internal knowledge to answer the additional questions generated
    to address gaps in the initial conclusion.
    """
    logging.info("--- Node: answer_additional_questions_with_llm ---")
    additional_questions = state.get('additional_questions', [])
    
    if not additional_questions:
        logging.warning("No additional questions found to answer.")
        return {"additional_answers": []}
    
    answered_questions: List[LLMAnsweredQuestion] = []
    
    for question in additional_questions:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert researcher with broad knowledge. For the given research question, "
             "provide a comprehensive answer based on your internal knowledge. This question was "
             "specifically generated to address gaps in understanding of the research topic, so focus "
             "on providing detailed, specific information that will help fill those gaps. "
             "If applicable, include relevant details, examples, or sub-aspects within your response."
            ),
            ("human", "Research Question: {question}")
        ])
        
        try:
            answer_chain = prompt | llm
            response = answer_chain.invoke({"question": question})
            answered_questions.append({
                "question": question,
                "comprehensive_answer": response.content
            })
            logging.info(f"LLM answered additional question: '{question}'")
        except Exception as e:
            logging.error(f"Error answering additional question '{question}' with LLM: {e}", exc_info=True)
            answered_questions.append({
                "question": question,
                "comprehensive_answer": f"Error generating answer for this question: {e}"
            })
    
    logging.info(f"Completed answering {len(answered_questions)} additional questions.")
    return {"additional_answers": answered_questions}
