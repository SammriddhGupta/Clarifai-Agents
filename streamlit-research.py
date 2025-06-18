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
    backstory=
                """You are a meticulous and insightful research analyst at a tech think tank.
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

st.sidebar.header("Agent & LLM Settings")
env_mode = st.sidebar.checkbox("Use CLARIFAI_PAT from environment", True)
if not env_mode:
    custom_key = st.sidebar.text_input(
        "Enter Clarifai PAT", type="password",
        help="Paste your Clarifai Personal Access Token here"
    )
model_name = st.sidebar.text_input("Model", clarifai_llm.model)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

no_limit = st.sidebar.checkbox("Use default max tokens (recommended)", False)
if not no_limit:
    max_tokens = st.sidebar.number_input(
        "Max Tokens", min_value=64, value=5056, step=64
    )
else:
    max_tokens = None

verbose = st.sidebar.checkbox("Verbose Output", True)
allow_delegation = st.sidebar.checkbox("Allow Delegation", False)

api_key = os.environ.get("CLARIFAI_PAT") if env_mode else custom_key
custom_clarifai_llm = LLM(
    model=model_name,
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=api_key,
    temperature=temperature,
    max_tokens=max_tokens
)

researcher.llm = custom_clarifai_llm
researcher.verbose = verbose
researcher.allow_delegation = allow_delegation

st.title("AI Research Analyst Agent")

st.write("Enter a topic below to have the AI agent research and generate a detailed report.")

topic = st.text_input("Research Topic", "")

can_run = env_mode or (not env_mode and custom_key.strip() != "")
if not can_run:
    st.sidebar.error("Please enter a valid Clarifai PAT or use the existing environment key.")

run_btn = st.button("Run Research", disabled=not can_run)
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner(f"Researching '{topic}'..."):
            report = run_research(topic.strip())
        st.subheader("Research Report")
        st.markdown(report)

st.sidebar.markdown("---")
st.markdown(
    """
    <style>
    .credits {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 65%;
        text-align: center;
        font-size: 1.1em
    }
    </style>
    <div class="credits">
        Built with <a href="https://www.clarifai.com/" target="_blank">Clarifai</a> and <a href="https://www.crewai.com/" target="_blank">CrewAI</a>
    </div>
    """,
    unsafe_allow_html=True
)