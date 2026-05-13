#//Dependency
#pip install langchain langchain-core langchain-groq python-dotenv


from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.0
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic", "tone", "target_audience"],
    template="""
You are an expert LinkedIn content writer.

Write a professional and engaging LinkedIn post based on the following details:

Topic: {topic}
Tone: {tone}
Target Audience: {target_audience}

Constarints:
- Words limit: 150-200 words
- Start with unique and catchy hook
- Looks like Professional linkedIn post
- Use 3-4 paragraphs only
- Use simple and engaging language
- Add appropriate emojis also
- End with relevant hashtags
"""
)

# User Input
topic = input("Enter Topic: ")
tone = input("Tone of post: ")
target_audience = input("Target Audience: ")

# Create Final Prompt
final_prompt = prompt.format(
    topic=topic,
    tone=tone,
    target_audience=target_audience
)

# Generate Response
response = llm.invoke(final_prompt)

# Print Output
print("\nGenerated LinkedIn Post:\n")
print(response.content)