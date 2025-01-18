import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent
from crewai_tools import SerperDevTool
from crewai import Task
from crewai import Crew, Process

# Load variables from .env
load_dotenv()

# Access the environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["SERPER_API_KEY"] = SERPER_API_KEY  
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Specify the model
os.environ["OPENAI_MODEL"] = "gpt-4-32k"

# Create the search tool
search_tool = SerperDevTool()

# Creating the research agent
researcher = Agent(
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies in {topic}",
    backstory=(
        "You're at the forefront of innovation, eager to explore and "
        "share knowledge that could change the world."
    ),
    memory=True,
    allow_delegation=True,
    tools=[search_tool]
)

# Creating the writer agent
writer = Agent(
    role="writer",
    goal="Narrate compelling tech stories about {topic}",
    backstory=(
        "You have a talent for breaking down complex ideas into simple, "
        "compelling stories that inform and engage, "
        "making new discoveries easy to understand and appreciate."
    )
)

# Define the research task
research_task = Task(
    description=(
        "Identify the next big trend in {topic}. "
        "Focus on identifying pros and cons and the overall narrative. "
        "Your final report should clearly articulate the key points, "
        "its market opportunities, and potential risks."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
    tools=[search_tool],
    agent=researcher,
)

# Define the writer task
writer_task = Task(
    description=(
        "Compose an insightful article on {topic}. "
        "Focus on the latest trends and how it's impacting the industry. "
        "This article should be easy to understand, engaging, and positive."
    ),
    expected_output="A 4 paragraph article on {topic} advancements formatted as markdown.",
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file="new-blog-post.md",
)

# Combine tasks into a crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    process=Process.sequential
)

# Streamlit interface
st.title("CrewAI Topic Research and Writing App")
st.write(
    "This app uses CrewAI to uncover groundbreaking technologies and narrate compelling stories. "
    "Enter a topic to get started!"
)

# Get user input
topic = st.text_input("Enter the topic:", "")

if st.button("Generate Report and Article"):
    if topic:
        # Run the crew process
        with st.spinner("Processing... Please wait."):
            result = crew.kickoff(inputs={'topic': topic})
        
        # Debugging: Display the raw result
        st.write("Raw result:", result)

        # Display results based on response type
        if isinstance(result, str):
            st.write("### Generated Output")
            st.write(result)
        elif isinstance(result, dict) and 'tasks' in result:
            # Ensure the expected structure exists
            st.write("### Research Report")
            st.write(result['tasks'][0]['output'])  # Researcher output
            st.write("### Written Article")
            st.markdown(result['tasks'][1]['output'])  # Writer output
        else:
            st.error("Unexpected response format. Please check the output.")
    else:
        st.error("Please enter a topic to proceed.")

