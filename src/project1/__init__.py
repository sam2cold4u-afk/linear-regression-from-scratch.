import os
import warnings
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress minor API warning logs
warnings.filterwarnings("ignore")

os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6Kg_CpPXbrffHpHUeax0mQP6D8glLJbXFPBPBJUGVUjng"


@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers."""
    print("\n[Tool called: calculator]")
    return f"The sum of {a} and {b} is {a + b}"


@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user."""
    print("\n[Tool called: say_hello]")
    return f"Hello {name}, I hope you are well today"


def main():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=os.environ["GEMINI_API_KEY"],
    )

    tools = [calculator, say_hello]
    agent_executor = create_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=user_input)]}
        )
        content = response["messages"][-1].content
        if isinstance(content, list):
            # Extract text elements if content is a list of blocks
            text_blocks = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
            print(" ".join(text_blocks))
        else:
            print(content)


if __name__ == "__main__":
    main()