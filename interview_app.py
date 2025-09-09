import streamlit as st
from typing import List, Dict, Tuple
import re
from pydantic import BaseModel
from openai import OpenAI
from helper_functions import *

question_count = 5
use_AI = False

class Questions(BaseModel):
    questions: List[str]

default_position = "Software Engineer"

DEFAULT_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want this job?",
    "Which are the programming languages you are most confortable with?",
    "What is the 3rd normalization form?",
    "Tell me about frameworks you have used in the past."
]

my_api_key = get_openai_api_key()
client = OpenAI(api_key=my_api_key)


def generate_questions(applied_for_position: str, question_count: int)->List[str]:
    if not use_AI:
        return DEFAULT_QUESTIONS

    BEHAVIORAL_COUNT = question_count*0.4
    TECHNICAL_COUNT = question_count - BEHAVIORAL_COUNT

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
        text_format=Questions
    )
    
    questions: List[str] = response.output_parsed.questions
    return questions

def generate_feedback(questions: List[str], answers: List[str]) -> List[str]:    
    if not use_AI:
        return ["No feedback possible without AI"] * len(questions)

    feedback = []

    for q, a in zip(questions, answers):
        response = client.responses.parse(
            model="gpt-4o",
            input=[
                {"role": "system", "content": f"You are an expert hiring manager providing feedback on interview answers."},
                {"role": "user", "content": f"""Task: Provide constructive feedback on the following interview answer.
                    Question: {q}
                    Answer: {a}
                    
                    Guidelines:
                    - Start with a positive note.
                    - Highlight 2-3 strengths in the answer.
                    - Suggest 2-3 specific areas for improvement.
                    - Be concise and professional.
                    
                    Example output:
                    ```
                    Positive: Great enthusiasm and clear communication.
                    Strengths: Strong problem-solving skills, relevant experience, good cultural fit.
                    Improvements: Provide more specific examples, quantify achievements, avoid filler words.
                    ```
                    
                    Provide the feedback in a similar structured format."""}
            ],
            temperature=1.0,
            top_p=0.9,
            max_output_tokens=300,
            text_format=str
        )
        feedback.append(response.output_parsed)

    return feedback
    

############################## MAIN ##############################
st.set_page_config(page_title="Interview Simulator", page_icon="🎤", layout="centered")

# -----------------------------
# Initialize the UI State if not done yet
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "step" not in st.session_state:
    st.session_state.step = 0 

if "applied_for_position" not in st.session_state:
    st.session_state.applied_for_position = default_position

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "finished" not in st.session_state:
    st.session_state.finished = False

if "show_results" not in st.session_state:
    st.session_state.show_results = False

st.title("🎤 Interview Simulator")

# -----------------------------
# Interview Flow
# -----------------------------
step = st.session_state.step

# Choose position and generate questions
if step == 0:
    st.session_state.applied_for_position = st.text_input("Position you are applying for:", value=default_position, key="input_applied_for_position")
    if st.button("Generate Questions"):
        st.session_state.questions = generate_questions(st.session_state.applied_for_position, question_count)
        st.session_state.step += 1
        st.session_state.answers = {}
        st.session_state.finished = False
        st.rerun()
else:
    # Move through questions
    st.caption(f"Answer {question_count} interview questions for the position: {st.session_state.applied_for_position}")

    if not st.session_state.finished or st.session_state.show_results:
        q = safe_get(st.session_state.questions, step-1, "No question found")
        st.subheader(f"Question {step}/{question_count}")
        st.markdown(f"**{q}**")
        saved_answer = safe_get(st.session_state.answers, step-1, "")
        
        if st.session_state.show_results:
            feedback = safe_get(st.session_state.feedback, step-1, "")
            st.markdown(f"**Your answer:**\n\n{saved_answer}")
            st.markdown(f"**Feedback:**\n\n{feedback}")
        else:
            ans = st.text_area("Your answer:", value=saved_answer, height=180, key=f"ans_{step-1}")

        cols = st.columns([1,1,1])

        if step > 1 and cols[1].button("← Previous"):
            if not st.session_state.show_results:
                st.session_state.answers[step-1] = ans
            st.session_state.step -= 1
            st.rerun()
        if step < question_count:
            if cols[2].button("Next →"):
                if not st.session_state.show_results:
                    st.session_state.answers[step-1] = ans
                st.session_state.step += 1
                st.rerun()
        else:
            if not st.session_state.show_results:
                if cols[2].button("Finish✅"):
                    # st.session_state.answers[step-1] = ans
                    st.session_state.finished = True
                    st.rerun()

        if st.session_state.show_results:
            if cols[2].button("Start over"):
                st.session_state.step = 0
                st.session_state.finished = False
                st.session_state.show_results = False
                st.session_state.questions = {}
                st.session_state.answers = {}
                st.rerun()
    else:
        # Finished - show results
        st.session_state.show_results = True
        st.session_state.feedback = generate_feedback(st.session_state.questions, st.session_state.answers)
        st.session_state.step = 1
        st.rerun()        
