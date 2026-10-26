"""Pretend customer: the six moods (personas in state.py).

Each mood has a description for the AI model, a patience limit enforced by code
(after this many messages without a solution, the customer hangs up), and the
words said when hanging up.
"""

from dataclasses import dataclass

from callverse.state import Persona


@dataclass(frozen=True)
class Mood:
    description: str
    patience: int           # client messages before hanging up if nothing is solved
    hang_up: str            # what the customer says when patience runs out


PERSONAS: dict[Persona, Mood] = {
    "calm": Mood(
        "Vous êtes calme et poli. Vous expliquez clairement votre situation et vous écoutez les réponses. "
        "Vous acceptez une solution raisonnable.",
        patience=8,
        hang_up="Je vois que ça n'avance pas. Je rappellerai plus tard, au revoir.",
    ),
    "impatient": Mood(
        "Vous êtes pressé. Vous faites des phrases courtes, vous voulez une réponse tout de suite et vous "
        "supportez mal qu'on vous fasse répéter ou attendre.",
        patience=4,
        hang_up="Bon, je n'ai pas le temps pour ça. Au revoir.",
    ),
    "dissatisfied": Mood(
        "Vous êtes mécontent de CallVerse et vous le dites. Vous restez correct, mais vous voulez que l'on "
        "reconnaisse le problème avant d'écouter une solution.",
        patience=6,
        hang_up="C'est vraiment décevant. Je raccroche.",
    ),
    "demanding": Mood(
        "Vous êtes exigeant. Vous voulez des réponses précises (montants, délais) et vous demandez un geste "
        "commercial. Vous insistez si la première réponse ne vous convient pas.",
        patience=7,
        hang_up="Je n'obtiens rien de concret. J'écrirai au service client. Au revoir.",
    ),
    "interested_in_offer": Mood(
        "Vous êtes curieux et ouvert. En plus de votre demande, vous voulez savoir s'il existe une offre "
        "plus intéressante pour vous, et vous posez des questions sur les prix.",
        patience=8,
        hang_up="Je vais réfléchir de mon côté. Merci, au revoir.",
    ),
    "likely_to_churn": Mood(
        "Vous envisagez sérieusement de quitter CallVerse pour un autre opérateur. Vous le dites pendant "
        "l'appel et vous comparez avec la concurrence. Il faut une vraie réponse pour vous faire rester.",
        patience=5,
        hang_up="Tant pis, je vais voir ailleurs. Au revoir.",
    ),
}
