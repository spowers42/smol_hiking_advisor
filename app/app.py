import gradio as gr

from app import constants
from app.agent import get_agent, parse_assistant_response

agent_executor = get_agent()


def respond(message, history):
    result = agent_executor.invoke({"messages": [("human", message)]})
    return parse_assistant_response(result["messages"][-1].content)


with gr.Blocks(
    title=constants.APP_TITLE,
    fill_height=True,
) as demo:
    gr.ChatInterface(
        fn=respond,
        title=constants.APP_TITLE,
        description=constants.APP_DESCRIPTION,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue=constants.APP_THEME_COLOR))
