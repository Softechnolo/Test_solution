import os
import openai

class OpenAI_API:
  def __init__(self, region="us-east-1"):  # Set default region to us-east-1
    openai.api_key = os.getenv('sk-proj-QJqjY8hUuT1cxK8kNRydT3BlbkFJOw1sMlbuNxaVOrkgLZMG')
    self.region = region

  def get_answer(self, prompt, engine="text-embedding-ada-002", temperature=0.5, max_tokens=100):
    try:
      response = openai.Completion.create(
          engine=engine,
          prompt=prompt,
          temperature=temperature,
          max_tokens=max_tokens,
          # Add region argument to the request
          region=self.region
      )
      return response.choices[0].text.strip()
    except Exception as e:
      print(f"Failed to get answer from OpenAI API: {e}")
      return None

