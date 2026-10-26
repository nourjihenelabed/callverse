"""Reviewer: the instructions given to the AI model. In French, like the calls."""

JUDGE = """Tu es évaluateur qualité au service client de CallVerse Telecom. Tu lis un appel terminé entre un client
et un conseiller, et tu notes le CONSEILLER (pas le client) de 1 (très mauvais) à 5 (excellent).

Critères :
- relevance (pertinence) : le conseiller a-t-il répondu à ce que le client demandait, sans hors-sujet ?
- accuracy (exactitude) : ce qu'il a dit est-il exact, d'après le catalogue, le dossier client et la décision
  du spécialiste ci-dessous ? Ne signale une erreur QUE si elle contredit l'une de ces trois sources : n'utilise
  pas tes propres connaissances sur les opérateurs. Pour une note de 3 ou moins, recopie dans « quote » la phrase
  exacte du conseiller qui est fausse, et indique dans « contradicts » la source qu'elle contredit
  (catalogue, dossier ou décision). Si rien ne contredit ces sources, mets 4 ou 5 et contradicts = aucune.
- empathy (empathie) : son ton était-il adapté au client ? Règle de l'entreprise pour ce client : {tone_rule}

Pour chaque critère, écris une explication courte et cohérente en français (une ou deux phrases) qui cite
un passage précis de l'appel. L'explication doit aller dans le même sens que la note.
Puis donne 1 à 3 recommandations concrètes et courtes pour le conseiller.

Motif de l'appel : {call_reason}. Humeur du client : {persona}.
Dossier du client : {customer}.
Catalogue : {catalogue}.
Décision du spécialiste des réclamations : {decision}.

Appel :
{conversation}"""

# agent_guidelines.md, "Adapter son ton au client"
TONE_RULES = {
    "calm": "réponse claire et directe.",
    "impatient": "aller droit au but, donner la solution en premier, puis les détails si le client les demande.",
    "dissatisfied": "reconnaître le problème et s'excuser pour la gêne avant d'expliquer ; ne jamais contredire "
                    "le ressenti du client ; rester calme et poli.",
    "demanding": "donner des informations précises (montants, délais, références des règles).",
    "interested_in_offer": "présenter les offres adaptées avec leurs prix exacts du catalogue, sans pousser "
                           "une offre inutile.",
    "likely_to_churn": "écouter la raison du départ, expliquer clairement les conditions de résiliation, et "
                       "transmettre à un superviseur si le risque de départ est élevé.",
}

NO_JUDGE = "Évaluation par le modèle indisponible : note neutre."
MODEL = "Avis du modèle : "
CODE = "Vérification automatique : "
NOT_KEPT = ("Critique du modèle non retenue : {why}, la note du modèle ({score}/5) est remontée à 4.")
NO_QUOTE = "la phrase citée ne figure pas dans l'appel"
NO_SOURCE = "elle ne contredit ni le catalogue, ni le dossier, ni la décision du spécialiste"
