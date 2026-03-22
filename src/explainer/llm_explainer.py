import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def build_prompt(text: str, label: str, confidence: float) -> str:
    '''
    Building a prompt for an LLM
    We provide the text, the predicted class, and the models confidence score
    The LLM must explain why the text belongs to that class
    '''
    return f'''You are an expert in psycholinguistics and rhetoric analysis.

A text was classified as: {label} (confidence: {confidence:.0%})

Class definitions:
- rational_argument: logical reasoning, facts, evidence, sources
- emotional_manipulation: pressure on emotions, bypassing logic
- demagogy_tricks: false dilemmas, label switching, populism
- fear_appeal: threats, catastrophizing, enemy imagery
- authority_appeal: appeal to authority without evidence

Text to analyze:
\"\"\"{text}\"\"\"

Provide a concise analysis:
1. Why this text belongs to class "{label}"
2. Specific words/phrases that signal this (quote them)
3. What cognitive bias or vulnerability is being exploited
4. How a reader can protect themselves from this technique

Be specific and educational. Maximum 150 words.'''


def explain(text: str, label: str, confidence: float) -> str:
    '''
    We send a request, and recieve a response
    text        - source text
    label       - predicted class from our model
    confidence  - model confidence, ranging from 0 to 1
    '''

    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    prompt = build_prompt(text, label, confidence)

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return response.choices[0].message.content
