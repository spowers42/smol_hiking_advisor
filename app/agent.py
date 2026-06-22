from langchain.agents import create_agent


class SimpleReActAgent:
    def __init__(self, llm, tools, prompt, max_iterations=10):
        self._agent = create_agent(
            model=llm,
            tools=list(tools),
            system_prompt=prompt,
        )
        self.max_iterations = max_iterations

    def invoke(self, input_dict: dict) -> dict:
        return self._agent.invoke(
            input_dict,
            config={"recursion_limit": self.max_iterations * 2 + 5},
        )
