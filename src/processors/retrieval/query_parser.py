"""
Query Parser for RAG Retrieval Microservice
Handles query normalization, expansion, and intent detection
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.core.config import config
from src.core.document_models import RetrievalQuery, QueryType, QueryExpansion

logger = logging.getLogger(__name__)


class QueryParser:
    """Parser for user queries with normalization and expansion"""
    
    def __init__(self):
        """Initialize query parser"""
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "would", "you", "your", "this",
            "these", "they", "them", "their", "there", "then", "than"
        }
        
        # Common query patterns
        self.question_patterns = {
            r"^what\s+is": "definition",
            r"^what\s+are": "definition",
            r"^how\s+does": "process",
            r"^how\s+to": "instruction",
            r"^why\s+": "explanation",
            r"^when\s+": "temporal",
            r"^where\s+": "location",
            r"^who\s+": "person",
            r"^which\s+": "selection",
            r"^can\s+you": "capability",
            r"^could\s+you": "capability",
            r"^would\s+you": "capability"
        }
        
        # Intent keywords
        self.intent_keywords = {
            "definition": ["define", "definition", "meaning", "what is", "what are"],
            "explanation": ["explain", "why", "reason", "cause", "because"],
            "instruction": ["how to", "steps", "process", "method", "way"],
            "comparison": ["compare", "difference", "versus", "vs", "better"],
            "example": ["example", "instance", "case", "sample"],
            "summary": ["summary", "overview", "brief", "summarize"]
        }
    
    def parse_query(self, raw_query: str, **kwargs) -> RetrievalQuery:
        """
        Parse a raw query into a structured RetrievalQuery object
        
        Args:
            raw_query: Raw user query
            **kwargs: Additional query parameters
            
        Returns:
            RetrievalQuery object
        """
        try:
            # Normalize the query
            normalized_query = self._normalize_query(raw_query)
            
            # Detect query type
            query_type = self._detect_query_type(normalized_query)
            
            # Extract filters from query or kwargs
            filters = self._extract_filters(normalized_query, kwargs.get("filters", {}))
            
            # Create RetrievalQuery object
            retrieval_query = RetrievalQuery(
                query=normalized_query,
                query_type=query_type,
                filters=filters,
                top_k=kwargs.get("top_k", config.retrieval.top_k_default),
                similarity_threshold=kwargs.get("similarity_threshold", config.retrieval.similarity_threshold),
                enable_reranking=kwargs.get("enable_reranking", config.retrieval.enable_reranking),
                enable_query_expansion=kwargs.get("enable_query_expansion", config.retrieval.enable_query_expansion),
                session_id=kwargs.get("session_id"),
                user_id=kwargs.get("user_id")
            )
            
            logger.info(f"Parsed query: '{raw_query}' -> '{normalized_query}' (type: {query_type.value})")
            return retrieval_query
            
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            # Return basic query on error
            return RetrievalQuery(
                query=raw_query,
                query_type=QueryType.HYBRID,
                filters=kwargs.get("filters", {}),
                top_k=kwargs.get("top_k", config.retrieval.top_k_default)
            )
    
    def expand_query(self, query: RetrievalQuery) -> QueryExpansion:
        """
        Expand a query with synonyms and related terms
        
        Args:
            query: RetrievalQuery object
            
        Returns:
            QueryExpansion object
        """
        try:
            if not query.enable_query_expansion:
                return QueryExpansion(
                    original_query=query.query,
                    expanded_queries=[query.query],
                    synonyms=[],
                    related_terms=[]
                )
            
            # Extract key terms
            key_terms = self._extract_key_terms(query.query)
            
            # Generate synonyms
            synonyms = self._generate_synonyms(key_terms)
            
            # Generate related terms
            related_terms = self._generate_related_terms(key_terms)
            
            # Create expanded queries
            expanded_queries = self._create_expanded_queries(query.query, synonyms, related_terms)
            
            # Detect intent
            intent = self._detect_intent(query.query)
            
            expansion = QueryExpansion(
                original_query=query.query,
                expanded_queries=expanded_queries,
                synonyms=synonyms,
                related_terms=related_terms,
                intent=intent,
                confidence=0.8  # Placeholder confidence
            )
            
            logger.debug(f"Query expansion: {len(expanded_queries)} variants, {len(synonyms)} synonyms")
            return expansion
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return QueryExpansion(
                original_query=query.query,
                expanded_queries=[query.query],
                synonyms=[],
                related_terms=[]
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text"""
        try:
            # Convert to lowercase
            normalized = query.lower().strip()
            
            # Remove extra whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # Remove special characters but keep important ones
            normalized = re.sub(r'[^\w\s\?\!\.\,\-]', ' ', normalized)
            
            # Clean up multiple spaces
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Ensure minimum length
            if len(normalized) < config.retrieval.min_query_length:
                logger.warning(f"Query too short: '{normalized}'")
                return query  # Return original if too short
            
            # Truncate if too long
            if len(normalized) > config.retrieval.max_query_length:
                normalized = normalized[:config.retrieval.max_query_length]
                logger.warning(f"Query truncated to {config.retrieval.max_query_length} characters")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing query: {str(e)}")
            return query
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query"""
        try:
            query_lower = query.lower()
            
            # Check for question patterns
            for pattern, intent in self.question_patterns.items():
                if re.search(pattern, query_lower):
                    if intent in ["definition", "explanation"]:
                        return QueryType.SEMANTIC
                    elif intent in ["instruction", "process"]:
                        return QueryType.HYBRID
                    else:
                        return QueryType.HYBRID
            
            # Check for specific keywords
            if any(keyword in query_lower for keyword in ["define", "definition", "meaning"]):
                return QueryType.SEMANTIC
            
            if any(keyword in query_lower for keyword in ["how to", "steps", "process"]):
                return QueryType.HYBRID
            
            if any(keyword in query_lower for keyword in ["example", "instance", "case"]):
                return QueryType.SEMANTIC
            
            # Check query length and complexity
            words = query.split()
            if len(words) <= 2:
                return QueryType.KEYWORD
            elif len(words) <= 5:
                return QueryType.HYBRID
            else:
                return QueryType.SEMANTIC
                
        except Exception as e:
            logger.error(f"Error detecting query type: {str(e)}")
            return QueryType.HYBRID
    
    def _extract_filters(self, query: str, existing_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract filters from query text"""
        try:
            filters = existing_filters.copy()
            query_lower = query.lower()
            
            # Extract author filter
            author_match = re.search(r'author[:\s]+([a-zA-Z\s]+)', query_lower)
            if author_match:
                author = author_match.group(1).strip()
                filters["author"] = author
            
            # Extract date filters
            date_patterns = [
                r'from\s+(\d{4})',
                r'since\s+(\d{4})',
                r'after\s+(\d{4})',
                r'before\s+(\d{4})',
                r'until\s+(\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    year = int(match.group(1))
                    if "from" in pattern or "since" in pattern or "after" in pattern:
                        filters["date_from"] = datetime(year, 1, 1)
                    else:
                        filters["date_to"] = datetime(year, 12, 31)
            
            # Extract document type filter
            doc_type_patterns = {
                "pdf": r'\bpdf\b',
                "doc": r'\b(doc|docx)\b',
                "html": r'\bhtml\b',
                "text": r'\b(text|txt)\b'
            }
            
            for doc_type, pattern in doc_type_patterns.items():
                if re.search(pattern, query_lower):
                    filters["document_type"] = f".{doc_type}"
                    break
            
            # Extract section filter
            section_match = re.search(r'section[:\s]+([a-zA-Z\s]+)', query_lower)
            if section_match:
                section = section_match.group(1).strip()
                filters["section"] = section
            
            return filters
            
        except Exception as e:
            logger.error(f"Error extracting filters: {str(e)}")
            return existing_filters
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        try:
            # Remove stop words and extract meaningful terms
            words = query.lower().split()
            key_terms = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in key_terms:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)
            
            return unique_terms
            
        except Exception as e:
            logger.error(f"Error extracting key terms: {str(e)}")
            return query.split()
    
    def _generate_synonyms(self, terms: List[str]) -> List[str]:
        """Generate synonyms for terms (simplified implementation)"""
        try:
            # Simple synonym mapping (in production, use a proper thesaurus or embedding-based similarity)
            synonym_map = {
                "ai": ["artificial intelligence", "machine learning", "ml"],
                "ml": ["machine learning", "ai", "artificial intelligence"],
                "algorithm": ["method", "approach", "technique"],
                "model": ["system", "framework", "approach"],
                "data": ["information", "dataset", "records"],
                "analysis": ["examination", "study", "investigation"],
                "result": ["outcome", "finding", "conclusion"],
                "method": ["approach", "technique", "algorithm"],
                "system": ["framework", "model", "platform"],
                "user": ["person", "individual", "customer"],
                "interface": ["ui", "user interface", "gui"],
                "database": ["db", "data store", "repository"],
                "application": ["app", "software", "program"],
                "performance": ["efficiency", "speed", "optimization"],
                "security": ["safety", "protection", "privacy"]
            }
            
            synonyms = []
            for term in terms:
                if term in synonym_map:
                    synonyms.extend(synonym_map[term])
            
            return list(set(synonyms))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error generating synonyms: {str(e)}")
            return []
    
    def _generate_related_terms(self, terms: List[str]) -> List[str]:
        """Generate related terms (simplified implementation)"""
        try:
            # Simple related terms mapping
            related_map = {
                "ai": ["neural network", "deep learning", "nlp", "computer vision"],
                "ml": ["training", "prediction", "classification", "regression"],
                "algorithm": ["complexity", "optimization", "efficiency"],
                "data": ["preprocessing", "cleaning", "validation", "quality"],
                "analysis": ["statistics", "visualization", "insights"],
                "security": ["encryption", "authentication", "authorization"],
                "performance": ["benchmarking", "profiling", "monitoring"],
                "database": ["query", "indexing", "normalization", "sql"]
            }
            
            related_terms = []
            for term in terms:
                if term in related_map:
                    related_terms.extend(related_map[term])
            
            return list(set(related_terms))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error generating related terms: {str(e)}")
            return []
    
    def _create_expanded_queries(self, original_query: str, synonyms: List[str], related_terms: List[str]) -> List[str]:
        """Create expanded query variants"""
        try:
            expanded_queries = [original_query]
            
            # Add synonym-based expansions
            if synonyms:
                # Replace terms with synonyms
                for synonym in synonyms[:3]:  # Limit to top 3 synonyms
                    expanded_query = original_query
                    for term in original_query.split():
                        if term.lower() in synonym.lower():
                            expanded_query = expanded_query.replace(term, synonym, 1)
                            break
                    if expanded_query != original_query:
                        expanded_queries.append(expanded_query)
            
            # Add related terms as additional context
            if related_terms:
                for related_term in related_terms[:2]:  # Limit to top 2 related terms
                    expanded_query = f"{original_query} {related_term}"
                    expanded_queries.append(expanded_query)
            
            return expanded_queries[:5]  # Limit to 5 total variants
            
        except Exception as e:
            logger.error(f"Error creating expanded queries: {str(e)}")
            return [original_query]
    
    def _detect_intent(self, query: str) -> Optional[str]:
        """Detect user intent from query"""
        try:
            query_lower = query.lower()
            
            # Check for intent keywords
            for intent, keywords in self.intent_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    return intent
            
            # Check for question patterns
            for pattern, intent in self.question_patterns.items():
                if re.search(pattern, query_lower):
                    return intent
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            return None
    
    def validate_query(self, query: RetrievalQuery) -> Tuple[bool, Optional[str]]:
        """Validate a parsed query"""
        try:
            # Check query length
            if len(query.query) < config.retrieval.min_query_length:
                return False, f"Query too short (minimum {config.retrieval.min_query_length} characters)"
            
            if len(query.query) > config.retrieval.max_query_length:
                return False, f"Query too long (maximum {config.retrieval.max_query_length} characters)"
            
            # Check top_k
            if query.top_k < 1 or query.top_k > config.retrieval.top_k_max:
                return False, f"top_k must be between 1 and {config.retrieval.top_k_max}"
            
            # Check similarity threshold
            if query.similarity_threshold < 0.0 or query.similarity_threshold > 1.0:
                return False, "similarity_threshold must be between 0.0 and 1.0"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            return False, f"Validation error: {str(e)}"

