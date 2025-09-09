import streamlit as st
from typing import List, Dict, Tuple
import re
from pydantic import BaseModel
from openai import OpenAI
from helper_functions import *

question_count = 5
applied_for_position = "Software Engineer"
useAI = False
DEFAULT_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want this job?",
    "Describe a challenging problem you solved.",
    "What are your strengths and weaknesses?",
    "Where do you see yourself in 3 years?"
]

my_api_key = get_openai_api_key()
client = OpenAI(api_key=my_api_key)


def generate_questions(question_count: int)->List[str]:
    if not useAI:
        return DEFAULT_QUESTIONS

    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {"role": "system", "content": f"You are the hiring manager for the positon {applied_for_position} at a tech company."},
            {"role": "user", "content": f"""Task: Produce EXACTLY {question_count} refined interview questions for this position. 
                - Behavioral: {BEHAVIORAL_COUNT}
                - Technical: {TECHNICAL_COUNT}
                Examples of the output:
                ```
                Can you describe a challenging software project you worked on and how you handled the obstacles?
                What programming languages are you most proficient in, and how have you applied them in previous projects?
                ```
                
                Once you have the questions, think over each question and refine them to be more specific and challenging.
                Output only the refined questions."""}
        ],
        temperature=1.0,
        top_p=0.9,
        max_output_tokens=question_count*40,
        response_model=List[str]
    )
    
    questions: List[str] = response.output_parsed.questions
    return questions
    


st.set_page_config(page_title="Interview Simulator", page_icon="🎤", layout="centered")

# -----------------------------
# Initialize the UI State if not done yet
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "questions" not in st.session_state or st.session_state.questions == {}:
    st.session_state.questions = generate_questions(question_count)

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "finished" not in st.session_state:
    st.session_state.finished = False

st.title("🎤 Interview Simulator")
st.caption(f"Answer {question_count} interview questions. Your answers are saved locally in this session.")


# -----------------------------
# Interview Flow
# -----------------------------
step = st.session_state.step
question_number = step + 1
if not st.session_state.finished:
    q = st.session_state.questions[step]
    st.subheader(f"Question {question_number}/{question_count}")
    st.markdown(f"**{q}**")
    default_value = safe_get(st.session_state.answers, step, "")
    ans = st.text_area("Your answer:", value=default_value, height=180, key=f"ans_{step}")
    
    cols = st.columns([1,1,1])

    if st.session_state.step > 0 and cols[1].button("← Previous"):
        st.session_state.answers[step] = ans
        st.session_state.step -= 1
        st.rerun()
    if question_number < question_count:
        if cols[2].button("Next →"):
            st.session_state.answers[step] = ans
            st.session_state.step += 1
            st.rerun()
    else:
        if cols[2].button("Finish✅"):
            st.session_state.answers[step] = ans
            st.session_state.finished = True
            st.rerun()
else:
    col1, col2 = st.columns([1,1])
    if col1.button("Start over"):
        st.session_state.step = 0
        st.session_state.finished = False
        st.session_state.questions = {}
        st.session_state.answers = {}
        st.rerun()
