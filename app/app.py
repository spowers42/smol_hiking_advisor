import gradio as gr

from app import constants
from app.agent import parse_assistant_response
from app.config import get_llm
from app.models.tool_calling_model import ToolCallingChatModel
from app.runtime import AgentRuntime, RuntimeConfig
from app.tools.user_preferences import get_user_preferences, set_preferences
from app.tools.weather import connect as connect_weather_mcp
from app.tools.weather import get_weather_tools

connect_weather_mcp()
llm = ToolCallingChatModel(chat=get_llm())
tools = [get_user_preferences, *get_weather_tools()]
config = RuntimeConfig(llm=llm, tools=tools, prompt=constants.SYSTEM_PROMPT)
agent_executor = AgentRuntime(config)


def respond(message, history):
    result = agent_executor.invoke({"messages": [("human", message)]})
    return parse_assistant_response(result["messages"][-1].content)


def update_prefs(fitness: int, experience: int, group: str):
    set_preferences(fitness=fitness, experience=experience, group_size=group)


with gr.Blocks(
    title=constants.APP_TITLE,
    fill_height=True,
    css="#app-title { text-align: center; }",
) as demo:
    gr.Markdown(f"# {constants.APP_TITLE}", elem_id="app-title")
    gr.Markdown(constants.APP_DESCRIPTION)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            gr.ChatInterface(fn=respond)
        with gr.Column(scale=1, min_width=240):
            with gr.Accordion(label="Hiker Profile", open=True):
                fitness = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Fitness Level",
                    info="Beginner —————————— FKT Runner",
                )
                experience = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Experience Level",
                    info="Beginner —————————— Mountaineer",
                )
                group_size = gr.Radio(
                    choices=["Solo", "2–3", "3+"],
                    value="Solo",
                    label="Group Size",
                )

    for widget in (fitness, experience, group_size):
        widget.change(
            fn=update_prefs,
            inputs=[fitness, experience, group_size],
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue=constants.APP_THEME_COLOR))
