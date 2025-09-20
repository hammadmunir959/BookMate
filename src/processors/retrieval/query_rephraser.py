#!/usr/bin/env python3
"""
Query Rephrasing Module for Enhanced Retrieval

This module provides intelligent query rephrasing capabilities using LLM
to generate multiple query variations based on document summaries as context.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.core.service_manager import service_manager
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class RephrasedQuery:
    """Represents a rephrased query with metadata"""
    original_query: str
    rephrased_query: str
    reasoning: str
    confidence_score: float
    attempt_number: int


class QueryRephraser:
    """
    Intelligent query rephrasing using LLM and document context

    Generates multiple query variations to improve retrieval success rates
    """

    def __init__(self):
        self.llm_client: Optional[LLMClient] = None
        logger.info("QueryRephraser initialized (LLM client will be loaded on first use)")

    def _get_llm_client(self):
        """Get the LLM client, initializing it if necessary"""
        if self.llm_client is None:
            try:
                self.llm_client = service_manager.get_service('llm_client')
                logger.info("QueryRephraser LLM client loaded successfully")
            except Exception as e:
                logger.warning(f"LLM client not available for QueryRephraser: {e}")
                self.llm_client = None
        return self.llm_client

    def rephrase_query(
        self,
        original_query: str,
        document_summaries: List[str],
        max_variations: int = 3
    ) -> List[RephrasedQuery]:
        """
        Generate multiple rephrased queries using document summaries as context

        Args:
            original_query: The original user query
            document_summaries: List of document summaries for context
            max_variations: Maximum number of rephrased queries to generate

        Returns:
            List of RephrasedQuery objects with variations
        """
        logger.info(f"QueryRephraser: Called with query='{original_query}', summaries={len(document_summaries)}")

        llm_client = self._get_llm_client()
        if not llm_client:
            logger.warning("LLM client not available, returning empty rephrased queries")
            return []

        try:
            # Build context from document summaries
            context_text = self._build_document_context(document_summaries)

            # Generate rephrased queries
            rephrased_queries = self._generate_rephrased_queries(
                original_query, context_text, max_variations
            )

            logger.info(f"Generated {len(rephrased_queries)} rephrased queries for: '{original_query}'")
            return rephrased_queries

        except Exception as e:
            logger.error(f"Error rephrasing query '{original_query}': {e}")
            return []

    def _build_document_context(self, document_summaries: List[str]) -> str:
        """Build a structured context string from document summaries"""
        if not document_summaries:
            return "No document context available."

        context_parts = []
        for i, summary in enumerate(document_summaries, 1):
            context_parts.append(f"DOCUMENT {i}:\n{summary}")

        return "\n\n".join(context_parts)

    def _generate_rephrased_queries(
        self,
        original_query: str,
        context: str,
        max_variations: int
    ) -> List[RephrasedQuery]:
        """Generate rephrased queries using LLM"""

        # Create the rephrasing prompt
        prompt = self._create_rephrasing_prompt(original_query, context, max_variations)

        # Generate response from LLM
        llm_client = self._get_llm_client()
        if not llm_client:
            logger.warning("LLM client not available during query generation")
            return []

        response = llm_client.generate_text(
            prompt=prompt,
            max_tokens=800,
            temperature=0.3,  # Lower temperature for more focused rephrasing
            system_prompt=self._get_rephrasing_system_prompt()
        )

        if not response:
            logger.warning("LLM failed to generate rephrased queries")
            return []

        # Parse the response into RephrasedQuery objects
        return self._parse_rephrasing_response(response, original_query)

    def _create_rephrasing_prompt(
        self,
        original_query: str,
        context: str,
        max_variations: int
    ) -> str:
        """Create the prompt for query rephrasing"""

        return f"""ANALYZE the user's query in the context of the provided documents and generate {max_variations} optimized search queries.

=== ORIGINAL QUERY ===
"{original_query}"

=== DOCUMENT CONTEXT ===
{context}

=== OPTIMIZATION TASK ===
Create {max_variations} strategically rephrased queries that maximize information retrieval success. Each rephrasing should:

STRATEGY PRINCIPLES:
1. **Semantic Preservation**: Maintain the exact core meaning and intent
2. **Document Alignment**: Use terminology and concepts from the document context
3. **Search Optimization**: Choose wordings that match likely document structures
4. **Alternative Perspectives**: Consider different ways the same information might be expressed

REPHRASING STRATEGIES TO CONSIDER:
- **Synonym Variation**: Use different words with same meaning
- **Structural Changes**: Rephrase sentence structure while keeping meaning
- **Specificity Adjustment**: Make more/less specific based on document context
- **Terminology Matching**: Use terms that appear in the document summaries
- **Question Type Variation**: Change from direct questions to statements or vice versa

=== OUTPUT FORMAT ===
Provide exactly {max_variations} rephrased queries in this format:

1. "Rephrased query text" - STRATEGY: [Brief explanation of why this rephrasing helps] - EXPECTED IMPACT: [Why it might find better results]

=== EXAMPLES ===
1. "What information does this document contain?" - STRATEGY: Changed from "about" to "contain" and made more direct - EXPECTED IMPACT: More specific terminology might match document indexing
2. "Can you summarize the document content?" - STRATEGY: Converted to question format - EXPECTED IMPACT: Question format might align with how information is organized

Generate {max_variations} optimized queries now:"""

    def _get_rephrasing_system_prompt(self) -> str:
        """Get the enhanced system prompt for query rephrasing"""
        return """You are an expert information retrieval specialist with deep knowledge of document analysis and search optimization.

YOUR EXPERTISE:
1. **Query Analysis**: Understanding user intent and information needs
2. **Document Structure**: Recognizing how different documents organize information
3. **Search Optimization**: Crafting queries that maximize retrieval success
4. **Semantic Understanding**: Using synonyms, related terms, and alternative phrasings

CRITICAL PRINCIPLES:
- **Semantic Equivalence**: Keep core meaning while varying expression
- **Document Context Awareness**: Use provided summaries to guide rephrasing
- **Search Strategy**: Consider different search angles (exact terms, broader concepts, specific details)
- **Precision vs Recall**: Balance finding more results vs finding highly relevant results

OUTPUT REQUIREMENTS:
- Provide exactly the requested number of rephrased queries
- Each query must be semantically similar but differently worded
- Include clear reasoning for why each rephrasing might work better
- Consider the document context when suggesting alternatives"""

    def _parse_rephrasing_response(
        self,
        response: str,
        original_query: str
    ) -> List[RephrasedQuery]:
        """Parse the enhanced LLM response into RephrasedQuery objects"""

        rephrased_queries = []
        lines = response.strip().split('\n')

        current_query = None
        current_strategy = ""
        current_impact = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for numbered items (1., 2., 3., etc.)
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                # Save previous query if exists
                if current_query:
                    reasoning = f"Strategy: {current_strategy.strip()}. Expected Impact: {current_impact.strip()}"
                    rephrased_query = RephrasedQuery(
                        original_query=original_query,
                        rephrased_query=current_query.strip(),
                        reasoning=reasoning,
                        confidence_score=0.8,  # Default confidence
                        attempt_number=len(rephrased_queries) + 1
                    )
                    rephrased_queries.append(rephrased_query)

                # Parse new query
                try:
                    # Extract the query part before the first dash
                    parts = line.split(' - ', 1)
                    if len(parts) >= 1:
                        query_part = parts[0].strip()

                        # Remove the numbering
                        query_text = query_part[3:].strip()  # Remove "1. "

                        # Remove quotes if present
                        if query_text.startswith('"') and query_text.endswith('"'):
                            query_text = query_text[1:-1]

                        current_query = query_text
                        current_strategy = ""
                        current_impact = ""

                        # Parse strategy and impact if available
                        if len(parts) > 1:
                            remaining = parts[1]

                            # Look for STRATEGY and EXPECTED IMPACT
                            if 'STRATEGY:' in remaining:
                                strategy_part = remaining.split('STRATEGY:', 1)[1]
                                if 'EXPECTED IMPACT:' in strategy_part:
                                    strat_impact = strategy_part.split('EXPECTED IMPACT:', 1)
                                    current_strategy = strat_impact[0].strip()
                                    current_impact = strat_impact[1].strip()
                                else:
                                    current_strategy = strategy_part.strip()
                            elif 'Reasoning:' in remaining.lower():
                                current_strategy = remaining.replace('Reasoning:', '').strip()

                except Exception as e:
                    logger.warning(f"Error parsing rephrased query line '{line}': {e}")
                    current_query = None
                    continue

        # Save the last query if exists
        if current_query:
            reasoning = f"Strategy: {current_strategy.strip()}. Expected Impact: {current_impact.strip()}"
            rephrased_query = RephrasedQuery(
                original_query=original_query,
                rephrased_query=current_query.strip(),
                reasoning=reasoning,
                confidence_score=0.8,  # Default confidence
                attempt_number=len(rephrased_queries) + 1
            )
            rephrased_queries.append(rephrased_query)

        logger.info(f"Successfully parsed {len(rephrased_queries)} rephrased queries")
        return rephrased_queries

    def get_rephrasing_stats(self) -> Dict[str, Any]:
        """Get statistics about query rephrasing performance"""
        llm_client = self._get_llm_client()
        return {
            "llm_available": llm_client is not None,
            "rephraser_initialized": True
        }
