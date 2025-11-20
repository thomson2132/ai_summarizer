import logging
import chromadb
from src.config import CHROMADB_PATH, COLLECTION_NAME
from src.embeddings import embedding_model
from src.pdf_processor import chunk_text
from src.device_manager import device_manager

logger = logging.getLogger(__name__)

class ChromaDBHandler:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=CHROMADB_PATH)
            logger.info(f"ChromaDB initialized at {CHROMADB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def add_paper(self, text: str, paper_name: str):
        try:
            # Delete old collection to avoid conflicts
            try:
                self.client.delete_collection(name=COLLECTION_NAME)
                logger.info("Deleted old collection")
            except:
                pass
            
            collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            chunks = chunk_text(text)
            
            # Filter empty chunks and very short chunks
            chunks = [c for c in chunks if len(c.strip()) > 15]
            
            if not chunks:
                logger.error("No valid chunks created from text")
                return
            
            logger.info(f"Creating embeddings for {len(chunks)} chunks...")
            embeddings = embedding_model.embed_batch(chunks)
            
            ids = [f"{paper_name}_{i}" for i in range(len(chunks))]
            
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                metadatas=[
                    {
                        "source": paper_name,
                        "chunk_id": i,
                        "chunk_length": len(chunk.split()),
                        "is_metadata": self._is_metadata_chunk(chunk)
                    } 
                    for i, chunk in enumerate(chunks)
                ],
                documents=chunks
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            
            if device_manager.device.type == "cuda":
                import torch
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error adding paper to ChromaDB: {str(e)}")
            raise
    
    def _is_metadata_chunk(self, chunk: str) -> bool:
        """Check if chunk contains metadata (title, authors, abstract)"""
        metadata_keywords = ['abstract', 'keywords', 'author', 'authors', 'university', 
                            'affiliation', 'correspondence', 'received', 'accepted', 'citation']
        return any(keyword in chunk.lower() for keyword in metadata_keywords)
    
    def retrieve(self, query: str, k: int = 3):
        try:
            collection = self.client.get_collection(name=COLLECTION_NAME)
            
            logger.info(f"Querying for: {query[:60]}")
            
            # Check if this is a metadata query (title, authors, etc.)
            metadata_keywords = ['title', 'author', 'abstract', 'university', 'affiliation', 'email']
            is_metadata_query = any(kw in query.lower() for kw in metadata_keywords)
            
            # For metadata queries, prioritize early chunks (which contain title, authors, etc.)
            if is_metadata_query:
                # Get all documents
                all_docs = collection.get()
                if all_docs and all_docs.get("documents"):
                    # Sort by chunk_id to get early chunks first
                    doc_metadata = list(zip(
                        all_docs["documents"],
                        all_docs.get("metadatas", [{}] * len(all_docs["documents"]))
                    ))
                    
                    # Prioritize first 10 chunks (usually contain title, authors, abstract)
                    early_chunks = []
                    other_chunks = []
                    
                    for doc, meta in doc_metadata:
                        chunk_id = meta.get("chunk_id", 999)
                        if chunk_id < 10 or meta.get("is_metadata", False):
                            early_chunks.append(doc)
                        else:
                            other_chunks.append(doc)
                    
                    # Return early chunks + some others for context
                    docs = early_chunks[:k] + other_chunks[:max(1, k-len(early_chunks))]
                    logger.info(f"Retrieved {len(docs)} documents (metadata query mode)")
                    return docs[:k]
            
            # Regular semantic search for non-metadata queries
            query_embedding = embedding_model.embed_text(query)
            
            # Increase k to get more candidates
            k_candidates = min(k * 3, 15)
            
            # Query ChromaDB with larger k
            results = collection.query(
                query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
                n_results=k_candidates
            )
            
            # Extract documents and distances
            if results["documents"] and len(results["documents"]) > 0:
                docs = results["documents"][0]
                distances = results.get("distances", [0]*len(docs))[0] if results.get("distances") else [0]*len(docs)
                metadatas = results.get("metadatas", [{}]*len(docs))[0] if results.get("metadatas") else [{}]*len(docs)
                
                # Score documents: prefer shorter distance + metadata chunks for metadata queries
                scored_docs = []
                for doc, distance, metadata in zip(docs, distances, metadatas):
                    if len(doc.strip()) < 15:
                        continue
                    
                    # Lower distance is better (inverse similarity)
                    score = distance
                    
                    # Boost metadata chunks
                    if metadata.get("is_metadata", False):
                        score *= 0.6  # Reduce distance (higher relevance)
                    
                    scored_docs.append((doc, score))
                
                # Sort by score and take top k
                scored_docs.sort(key=lambda x: x[1])
                docs = [doc for doc, _ in scored_docs[:k]]
            else:
                docs = []
            
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        
        except Exception as e:
            logger.error(f"Error retrieving from ChromaDB: {str(e)}")
            return []

chroma_handler = ChromaDBHandler()
