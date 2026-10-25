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


# ---------------------------------------------------------------- the pretend customer

FAKE_FACTS = {"codes": ["box_fibre", "forfait_mobile"], "offers": "Box Fibre, Forfait Mobile 100 Go",
              "bill": "49,98 €", "double_bill": "99,96 €", "box": "Box Fibre", "contract": "sans engagement",
              "payment": "prélèvement automatique", "tenure": 14, "other_offer": "Box ADSL",
              "other_option": "TV", "churn_risk": 0.4}


class StructuredScript:
    """A fake model whose structured answers are given in advance, one per call."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.sent = []

    def with_structured_output(self, schema):
        model = self

        class Structured:
            def invoke(self, prompt):
                model.sent.append(list(prompt) if isinstance(prompt, list) else prompt)
                answer = model.answers.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return schema(**answer)

        return Structured()


class PoliteCustomer:
    """For the whole-call tests: opens with the stub's sentences, then accepts the first answer."""

    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, prompt):
                if schema.__name__ == "Opening":
                    complaint = "réclamation" in str(prompt)
                    return schema(message="Bonjour, ma facture est beaucoup trop élevée ce mois-ci, ce n'est pas normal."
                                  if complaint else "Bonjour, je voudrais comprendre ma dernière facture.")
                return schema(decision="accept", message="D'accord, merci beaucoup. Au revoir.", satisfaction=4)

        return Structured()
