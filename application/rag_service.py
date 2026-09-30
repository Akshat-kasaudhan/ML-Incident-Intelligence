import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self, collection_name: str = "runbooks"):
        # Initialize embedding model (using a small, fast model for demo purposes)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize Qdrant Client (in-memory for demo, but can point to a real URL)
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        
        # Create collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
        # Seed with some mock data
        self._seed_mock_runbooks()
        
    def _seed_mock_runbooks(self):
        """Seeds the vector DB with some mock runbooks for our known services."""
        runbooks = [
            {
                "service": "currencyservice",
                "title": "High Memory Usage Troubleshooting",
                "content": "When currencyservice exhibits high memory usage, immediately check the Redis cache connection. If the connection is stalled, it buffers requests in memory. Resolution: Restart the currencyservice pod and verify Redis connectivity."
            },
            {
                "service": "recommendationservice",
                "title": "Disk I/O Saturation",
                "content": "If recommendationservice is hitting 100% disk I/O, it's usually due to a stalled model weights download. Resolution: Scale up the disk IOPS dynamically via AWS console, or flush the local cache directory."
            },
            {
                "service": "emailservice",
                "title": "TCP Retransmissions and Drop Offs",
                "content": "Email service network drops are often caused by exhaustion of the NAT gateway ports. Resolution: Scale out the email service to more nodes to distribute the TCP connections, or check the SendGrid API limits."
            }
        ]
        
        points = []
        for idx, rb in enumerate(runbooks):
            vector = self.encoder.encode(rb["content"]).tolist()
            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=vector,
                    payload={"service": rb["service"], "title": rb["title"], "content": rb["content"]}
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def query_runbooks(self, query_text: str, top_k: int = 2) -> List[Dict]:
        """Query the vector database for relevant runbooks."""
        query_vector = self.encoder.encode(query_text).tolist()
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score,
                "title": hit.payload["title"],
                "content": hit.payload["content"],
                "service": hit.payload["service"]
            })
            
        return results

    def generate_rca_summary(self, predicted_service: str, evidence: List[str]) -> str:
        """
        In a real scenario, this would use LangChain to prompt an LLM (like GPT-4 or Gemini)
        to synthesize the evidence and the retrieved runbook into a coherent RCA report.
        """
        query = f"How to fix {predicted_service} issues? Evidence: {' '.join(evidence)}"
        relevant_docs = self.query_runbooks(query)
        
        if not relevant_docs:
            return "No relevant runbooks found for this issue."
            
        top_doc = relevant_docs[0]
        
        # Mocking the LLM generation for now
        summary = f"**RAG Augmented Summary:**\n\n"
        summary += f"Based on the evidence (`{evidence[0]}`), the system strongly implicates **{predicted_service}**.\n\n"
        summary += f"**Historical Knowledge Base (Runbook: {top_doc['title']}):**\n"
        summary += f"> {top_doc['content']}\n\n"
        summary += f"**Recommended Action:** Follow the steps in the runbook above."
        
        return summary

# Instantiate a singleton for the app
rag_client = RAGService()
