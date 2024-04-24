from pinecone import Pinecone, ServerlessSpec

class VectorDatabase:
  def __init__(self, index_name="mohammed", region="us-east-1"):  # Set default region
    self.index_name = index_name.lower().replace(" ", "-")
    pc = Pinecone(api_key="ff9ec1d1-de35-45a4-bab7-cba9f33dbe4e")

    # Check if index exists, create it only if necessary (assuming free tier allows creation)
    if self.index_name not in pc.list_indexes().names():
      try:
        pc.create_index(
            name=self.index_name,
            dimension=384,  # As per the project requirement Sir
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=region)  # Use provided region argument
        )
      except pinecone.core.client.exceptions.PineconeApiException as e:
        if e.code == "INVALID_ARGUMENT":  # Handle specific error for region limitation
          print(f"Error creating index: {e.message}. Consider using a supported free tier region.")
        else:
          raise e  # Re-raise other exceptions

    # Directly return the created Pinecone Index object (best approach)
    self.index = pc.Index(name=self.index_name)  # Pass name explicitly

  def addData(self, id, vector):
    self.index.upsert(items={id: vector})

  def find_best_matches(self, query_vector, top_k=5):
    results = self.index.query(queries=query_vector, top_k=top_k)
    return results.ids[0]
