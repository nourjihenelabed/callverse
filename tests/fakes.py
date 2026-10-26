"""A fake AI model for the tests, so they run without Ollama.

It plays a script: each call to invoke() returns the next prepared reply.
It records what it was sent, so a test can check what the worker told the model.
"""

from langchain_core.messages import AIMessage


def tool_call(name: str, call_id: str = "call-1", **args) -> AIMessage:
    """A model reply that asks to use a tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ScriptedModel:
    def __init__(self, replies=(), intent="check_invoice", is_complaint=False):
        self.replies = [AIMessage(r) if isinstance(r, str) else r for r in replies]
        self.understanding = {"intent": intent, "is_complaint": is_complaint}
        self.sent = []              # every list of messages given to invoke()
        self.tool_names = []

    # the same three methods the real ChatOllama model offers
    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        return self

    def with_structured_output(self, schema):
        model = self

        class Structured:
            def invoke(self, prompt):
                model.sent.append(prompt)
                return schema(**model.understanding)

        return Structured()

    def invoke(self, messages):
        self.sent.append(list(messages) if isinstance(messages, list) else messages)   # a copy, as sent
        if not self.replies:
            return AIMessage("Réponse par défaut. Puis-je faire autre chose pour vous ?")
        return self.replies.pop(0)


class KeywordModel(ScriptedModel):
    """For the whole-call tests: a complaint when the customer says it is not normal."""

    def with_structured_output(self, schema):
        model = self

        class Structured:
            def invoke(self, prompt):
                last = prompt.strip().splitlines()[-1].lower()
                complaint = "pas normal" in last or "inadmissible" in last
                return schema(intent="complaint_billing" if complaint else "check_invoice", is_complaint=complaint)

        return Structured()
