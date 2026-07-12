"""
research/pdf_reader.py
Reads a PDF, chunks the text, and saves it into Charlie's RAG memory.
"""
import fitz  # PyMuPDF
import logging

logger = logging.getLogger("Charlie.pdf")

class PDFReader:
    def __init__(self, memory):
        self.memory = memory

    def ingest_pdf(self, file_path: str):
        """Reads a PDF or TXT and saves its chunks into RAG memory."""
        try:
            if file_path.endswith(".txt"):
                return self._ingest_txt(file_path)
                
            doc = fitz.open(file_path)
            # Try to get the real title, fallback to the file name
            title = doc.metadata.get("title") or file_path.split("/")[-1]
            
            chunk_size = 500 # Split text into 500-character blocks
            chunks_added = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                
                if not text:
                    continue
                    
                # ── The Chunking Logic ──
                # Slice the page string into a list of 500-char strings
                chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    # Create a unique ID so we don't save duplicates if you run it twice
                    chunk_id = f"{title}_page{page_num}_chunk{i}"
                    
                    # Push it to ChromaDB
                    self.memory.collection.upsert(
                        documents=[f"Excerpt from {title}: {chunk}"],
                        metadatas=[{"source": title, "page": page_num}],
                        ids=[chunk_id]
                    )
                    chunks_added += 1
                    
            logger.info("Successfully ingested PDF: %s (%d chunks added)", title, chunks_added)
            return True
            
        except Exception as e:
            logger.error("Failed to read PDF %s: %s", file_path, e)
            return False

    def _ingest_txt(self, file_path: str):
        """Helper to ingest raw text files using the same chunking logic."""
        try:
            title = file_path.split("/")[-1].split("\\")[-1]
            chunk_size = 500
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                
            if not text:
                return False
                
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            chunks_added = 0
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{title}_chunk{i}"
                self.memory.collection.upsert(
                    documents=[f"Excerpt from {title}: {chunk}"],
                    metadatas=[{"source": title}],
                    ids=[chunk_id]
                )
                chunks_added += 1
                
            logger.info("Successfully ingested TXT: %s (%d chunks added)", title, chunks_added)
            return True
        except Exception as e:
            logger.error("Failed to read TXT %s: %s", file_path, e)
            return False
