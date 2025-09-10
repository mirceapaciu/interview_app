import streamlit as st
from typing import List, Dict, Tuple
import re
from pydantic import BaseModel
from openai import OpenAI
from helper_functions import *

question_count = 5
use_AI = True
use_default_questions = True
use_default_answers = True

class Questions(BaseModel):
    questions: List[str]

default_job_title = "Software Engineer"

DEFAULT_QUESTIONS = [
    "Can you describe a challenging software project you worked on, detailing the specific obstacles you encountered and the strategies you used to overcome them?",
    "Tell me about a time you had to adapt to a significant change on a project. How did you manage your response and guide your team through it?",
    "What programming languages are you most proficient in, and can you provide specific examples of how you've used them to solve complex problems in past projects?",
    "Walk me through the process you would use to optimize a poorly performing piece of code. How do you identify the issues, and what tools or techniques do you employ?",
    "Explain a technical problem you've solved that required collaborating with other teams or departments. How did you ensure effective communication and integration across different areas?"
]

DEFAULT_ANSWERS = [
    """One of the most challenging projects I worked on was the migration of a large telecom billing system to a new architecture. The codebase had been developed over many years by different teams, with inconsistent standards and limited documentation.
The main obstacles were:
- Complex dependencies between legacy modules, making even small changes risky.
- Performance bottlenecks in database queries that slowed down batch processes.
- Limited stakeholder alignment, as different departments had conflicting priorities.

To overcome these, I:
- Mapped critical dependencies using static analysis and documentation workshops, which reduced the “black box” factor.
- Optimized database queries by introducing indexes, rewriting inefficient SQL, and caching results where possible. This improved batch processing times significantly.
- Improved communication by setting up regular cross-team syncs and a clear backlog, so priorities became transparent and conflicts could be resolved earlier.

The result was a stable migration with improved performance and clearer ownership of system components.""",

"""On one project, the client changed the core requirements mid-development, which affected the database schema and API contracts. 
I first analyzed the impact and created a revised implementation plan. I communicated the changes clearly to the team, re-prioritized tasks, and introduced short daily check-ins to track progress. 
This kept the team aligned and allowed us to deliver the updated system on schedule.""",

"""I’m most proficient in Java, C++, and SQL/PLSQL.
Java: Built a microservices-based telecom billing system, handling high-volume transactions and implementing REST APIs with Spring.
C++: Developed performance-critical modules for data processing in a telecom platform, optimizing memory usage and execution speed.
SQL/PLSQL: Designed complex stored procedures and optimized large-scale queries to improve batch processing times by 40%.""",

"""First, I profile the code to identify bottlenecks using tools like Java Flight Recorder, VisualVM, or SQL execution plans. 
Then I analyze algorithm efficiency, looking for high-complexity operations or unnecessary loops. 
I optimize by refactoring logic, caching results, reducing database calls, or using parallel processing where safe. 
Finally, I benchmark and test to ensure improvements don’t break functionality and achieve measurable performance gains.""",

"""In a telecom billing project, a new reporting feature required coordination between development, database, and operations teams. 
I organized joint requirement sessions to clarify dependencies, documented integration points, and used shared tickets and daily stand-ups to track progress. 
Clear communication and agreed-upon interfaces ensured smooth integration and on-time delivery."""
]

my_api_key = get_openai_api_key()
client = OpenAI(api_key=my_api_key)


def generate_questions(job_title: str, question_count: int)->List[str]:
    if use_default_questions:
        if use_default_answers:
            st.session_state.answers = DEFAULT_ANSWERS.copy()
        return DEFAULT_QUESTIONS

    BEHAVIORAL_COUNT = question_count*0.4
    TECHNICAL_COUNT = question_count - BEHAVIORAL_COUNT

    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {"role": "system", "content": f"You are the hiring manager for the positon {job_title} at a tech company."},
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
            max_output_tokens=300
        )
        feedback.append(response.output_text)

    return feedback
    
def is_valid_job_title(title: str) -> bool:
    #The job title should:\n- Be 3-50 characters long\n- Only contain letters, numbers, spaces, hyphens, and ampersands
    pattern = r'^[A-Za-z0-9 &-]{3,50}$'
    return bool(re.match(pattern, title))


############################## MAIN ##############################
st.set_page_config(page_title="Interview Simulator", page_icon="🎤", layout="centered")

# -----------------------------
# Initialize the UI State if not done yet
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "step" not in st.session_state:
    st.session_state.step = 0 

if "job_title" not in st.session_state:
    st.session_state.job_title = default_job_title

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

# Choose job_title and generate questions
if step == 0:
    job_title = st.text_input("Job title you are applying for:", value=default_job_title, key="input_job_title")

    # Always show the button in the same place
    generate_clicked = st.button("Generate Questions")

    if is_valid_job_title(job_title):
        st.session_state.job_title = job_title
        if generate_clicked:
            st.session_state.questions = generate_questions(st.session_state.job_title, question_count)
            st.session_state.step += 1
            st.session_state.finished = False
            st.rerun()
    else:
        st.error("The job title should:\n- Be 3-50 characters long\n- Only contain letters, numbers, spaces, hyphens, and ampersands")
else:
    # Move through questions
    st.caption(f"Answer {question_count} interview questions for the position: {st.session_state.job_title}")

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
