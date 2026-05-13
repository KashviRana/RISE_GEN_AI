#//Dependency
#pip install langchain langchain-core langchain-google-genai google-generativeai python-dotenv

from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["match_situation", "batsman", "bowler"],
    template="""
You are legendary cricket commentator.

Generate realistic and exciting live cricket commentary in Ravi Shastri's energetic and insightful style.

Match Situation: {match_situation}
Batsman on Strike: {batsman}
Bowler: {bowler}

Constraints:
- Words limit: 50-100 words
- Sound like live TV commentary
- Add excitement, suspense, and crowd reactions
- Mention batsman and bowler naturally
- Make it immersive and professional
"""
)

# User Input
match_situation = input("Situation of Match: ")
batsman = input("Batsman on Strike: ")
bowler = input("Name of Bowler: ")

# Create Final Prompt
final_prompt = prompt.format(
    match_situation=match_situation,
    batsman=batsman,
    bowler=bowler
)

# Generate Response
response = llm.invoke(final_prompt)

# Print Output
print("\nGenerated Cricket Commentary:\n")
print(response.content[0].get("text"))