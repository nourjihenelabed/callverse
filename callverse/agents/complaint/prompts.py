"""Complaint specialist: the instructions given to the AI model at each step."""

CLASSIFY = """Tu es le spécialiste des réclamations de CallVerse Telecom.
Lis la conversation et classe la réclamation du client.

Catégories (une seule, la plus proche du problème principal) :
- billing : montant trop élevé, option facturée sans accord, double prélèvement
- technical : coupures internet, débit trop lent, box en panne, pas de réseau mobile
- contract : changement d'offre non appliqué, conditions d'engagement contestées
- service_quality : attente trop longue, conseiller impoli, rappel promis non fait
- termination : frais de résiliation contestés, facturation après résiliation
- delivery : box non reçue, technicien absent au rendez-vous

Gravité :
1 = gêne mineure, sans perte d'argent ni de service
2 = perte d'argent ou de service limitée
3 = perte importante ou problème qui dure (coupure de plus de 48 h, erreur de plus de 50 €)
4 = situation critique (plus d'internet depuis plus de 7 jours, personne vulnérable, prélèvements répétés à tort)
En cas de doute entre deux niveaux, choisis le plus élevé.

legal_threat = true seulement si le client parle d'avocat, de procès, de plainte,
d'association de consommateurs ou du médiateur.

Motif de l'appel : {call_reason}

Conversation :
{conversation}
"""

DRAFT = """Tu es le spécialiste des réclamations de CallVerse Telecom.
Propose une solution à la réclamation ci-dessous. Tu ne parles pas au client :
le conseiller lui expliquera ta proposition.

Réclamation : catégorie {category}, gravité {severity}.
Réclamations précédentes du client dans la même catégorie (12 mois) : {repeat_count}.

Conversation :
{conversation}

Règles de l'entreprise (utilise seulement celles-ci) :
{rules}

Réponds avec :
- actions : 1 à 4 actions concrètes, courtes, en français, pour le conseiller.
- compensation_euros : le geste commercial en euros (5,85 € = 5.85), selon la grille.
  Un remboursement d'un montant facturé à tort n'est PAS une compensation : mets-le dans les actions.
  Mets 0 si la grille ne prévoit rien.
- rules_used : les numéros [n] des règles qui justifient ta décision.
"""

# Step 3 searches the rules in French: the category codes ("technical") mean little to the search.
SEARCH_TOPICS = {
    "billing": "une erreur de facturation, un montant facturé à tort ou un double prélèvement",
    "technical": "une coupure de service internet, une panne ou un débit trop lent",
    "contract": "un contrat, un changement d'offre non appliqué ou une durée d'engagement contestée",
    "termination": "une résiliation, des frais de résiliation contestés ou une facture après résiliation",
    "service_quality": "la qualité de service, un conseiller impoli, un rappel promis non fait ou une attente trop longue",
    "delivery": "la livraison et l'installation, une box non reçue ou un technicien absent au rendez-vous",
}

SEARCH_QUESTIONS = (
    "Quelle compensation accorder pour {topic} ?",
    "Que doit vérifier le spécialiste pour {topic} ?",
)
