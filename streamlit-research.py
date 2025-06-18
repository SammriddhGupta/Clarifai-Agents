import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

clarifai_llm = LLM(
    model="openai/deepseek-ai/deepseek-chat/models/DeepSeek-R1-Distill-Qwen-7B",
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ.get("CLARIFAI_PAT")
)

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments and facts on a given topic",
    backstory="""You are a meticulous and insightful research analyst at a tech think tank.
You specialize in identifying trends, gathering verified information,
and presenting concise insights.""",
    verbose=True,
    allow_delegation=False,
    llm=clarifai_llm
)

def create_research_task(topic):
    return Task(
        description=f"""Conduct a comprehensive analysis of '{topic}'.
Identify key trends, breakthrough technologies, important figures, and potential industry impacts.
Focus on factual and verifiable information.""",
        expected_output="A detailed analysis report in bullet points, including sources if possible.",
        agent=researcher
    )


def run_research(topic):
    task = create_research_task(topic)

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("AI Research Agent with Clarifai & CrewAI")

st.write("Enter a topic below to have the AI research agent generate a concise report.")

topic = st.text_input("Research Topic", "")

if st.button("Run Research"):
    if topic.strip():
        with st.spinner(f"Researching '{topic}'..."):
            result = run_research(topic.strip())
        st.subheader("Research Report")
        st.text(result)
    else:
        st.warning("Please enter a topic to research.")