class TextSplitter:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def split(self, text):
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        
        # Split the text into paragraphs
        paragraphs = text.split('\n')
        chunks = []
        
        for paragraph in paragraphs:
            # Split the paragraph into words
            words = paragraph.split()
            
            # Group the words into chunks and add them to the list
            paragraph_chunks = [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
            chunks.extend(paragraph_chunks)
        
        return chunks

