"""
handlers/pdf_handler.py
Scans the data/pdfs/ folder and uses the PDFReader to ingest them.
"""
import os
import logging
from research.pdf_reader import PDFReader

logger = logging.getLogger("Charlie.handler.pdf")

def handle(memory) -> str:
    """Finds all PDFs in the folder and ingests them into memory."""
    pdf_dir = "data/pdfs"
    
    if not os.path.exists(pdf_dir):
        return "I couldn't find the PDF folder."
        
    # Find all files ending with .pdf or .txt
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf") or f.endswith(".txt")]
    
    if not pdf_files:
        return "There are no PDF files in the folder right now."
        
    reader = PDFReader(memory)
    success_count = 0
    
    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        logger.info("Ingesting %s...", filename)
        if reader.ingest_pdf(filepath):
            success_count += 1
            
    if success_count == 0:
        return "I tried reading the PDFs, but ran into an error."
        
    if success_count == 1:
        return "I have successfully read the PDF document. You can ask me questions about it now."
        
    return f"I have finished reading {success_count} PDF documents. What would you like to know?"
