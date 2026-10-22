# Traitement des réclamations CallVerse Telecom

Ce document décrit comment le spécialiste des réclamations traite une réclamation : les catégories, l'échelle de gravité, les étapes obligatoires et les cas où un superviseur doit décider. Les montants de compensation sont dans `compensation_grid.md`.

## Qu'est-ce qu'une réclamation

Une réclamation est une demande où le client exprime un mécontentement sur un service, une facture ou un traitement reçu, et attend une correction ou un geste. Une simple question (« comment lire ma facture ? ») n'est pas une réclamation. Une question sur une anomalie (« ma facture a doublé, ce n'est pas normal ») est une réclamation.

## Catégories de réclamation

Chaque réclamation reçoit une seule catégorie, la plus proche du problème principal :

| Code | Catégorie | Exemples |
| --- | --- | --- |
| billing | Facturation | Montant trop élevé, option facturée sans accord, double prélèvement |
| technical | Technique | Coupures internet, débit trop lent, box en panne, pas de réseau mobile |
| contract | Contrat | Changement d'offre non appliqué, conditions d'engagement contestées |
| service_quality | Qualité de service | Attente trop longue, conseiller impoli, rappel promis non fait |
| termination | Résiliation | Frais de résiliation contestés, facturation après résiliation |
| delivery | Livraison et installation | Box non reçue, technicien absent au rendez-vous |

## Échelle de gravité

La gravité va de 1 à 4 :

| Gravité | Signification | Exemples |
| --- | --- | --- |
| 1 | Gêne mineure, sans perte d'argent ni de service | Facture difficile à comprendre, attente un peu longue |
| 2 | Perte d'argent ou de service limitée | Option facturée par erreur un mois, coupures ponctuelles |
| 3 | Perte importante ou problème qui dure | Coupure internet de plus de 48 heures, erreur de facture de plus de 50 € |
| 4 | Situation critique | Plus d'internet depuis plus de 7 jours, client en télétravail ou personne vulnérable sans service, prélèvements répétés à tort |

En cas de doute entre deux niveaux, choisir le niveau le plus élevé.

## Étapes obligatoires de traitement

Le spécialiste suit toujours ces étapes, dans cet ordre :

1. **Classifier** : choisir la catégorie et la gravité.
2. **Vérifier l'historique** : consulter les réclamations précédentes du client et compter celles de la même catégorie sur les 12 derniers mois, résolues ou non.
3. **Rechercher les règles** : consulter cette procédure, la grille de compensation et les documents utiles (contrats, procédures).
4. **Proposer une solution** : actions concrètes, compensation éventuelle, et règles qui justifient la décision.
5. **Vérifier la solution** : la compensation respecte les plafonds, aucune étape obligatoire n'est oubliée.
6. **Décider** : traiter directement ou transmettre à un superviseur (voir ci-dessous).
7. **Suivre** : créer ou mettre à jour le ticket de la réclamation.

Le spécialiste ne parle jamais directement au client. Il transmet sa proposition au conseiller, qui l'explique au client.

## Cas où un superviseur doit décider

La réclamation est obligatoirement transmise à un superviseur dans les cas suivants. Le conseiller ne peut pas les traiter seul :

1. **Menace juridique** : le client mentionne un avocat, un procès, une plainte, une association de consommateurs ou le médiateur.
2. **Réclamation répétée** : c'est au moins la 3e réclamation du client dans la même catégorie sur les 12 derniers mois.
3. **Risque élevé de résiliation** : l'indicateur interne de risque de départ du client est de 0,70 ou plus. Cet indicateur ne doit jamais être mentionné au client.
4. **Compensation au-delà du plafond** : la compensation nécessaire dépasse le plafond automatique de `compensation_grid.md`.

Dans ces cas, le conseiller explique au client que son dossier est transmis à un responsable, qui le recontacte sous 48 heures ouvrées. Le ticket passe à l'état « solution proposée » en attendant la décision.

## Étapes obligatoires par catégorie

- **Facturation** : vérifier la facture concernée et les produits du client avant toute proposition. Une erreur de facturation reconnue est toujours remboursée en totalité ; ce remboursement n'est pas une compensation et ne compte pas dans le plafond.
- **Technique** : vérifier s'il existe une panne connue dans la zone, puis lancer un diagnostic à distance avant de proposer une compensation. Un ticket technique doit être ouvert si la panne n'est pas résolue pendant l'appel.
- **Contrat** : vérifier le contrat signé et la date de début d'engagement avant de répondre.
- **Qualité de service** : présenter des excuses claires et noter le problème ; proposer un rappel à une heure choisie par le client si un rappel avait été promis.
- **Résiliation** : recalculer les frais selon `contracts.md` ; si des frais ont été facturés à tort, les rembourser en totalité.
- **Livraison** : vérifier le suivi du colis ou le rendez-vous technicien ; proposer un nouveau rendez-vous sous 72 heures.

## Suivi des tickets

Chaque réclamation a un ticket qui passe par ces états :

| État | Signification |
| --- | --- |
| open | Réclamation enregistrée, pas encore de solution |
| solution_proposed | Une solution a été proposée au client, ou le dossier attend un superviseur |
| accepted | Le client a accepté la solution |
| refused | Le client a refusé la solution |
| resolved | La solution est appliquée et le problème est réglé |
| reopened | Le même problème est revenu après la résolution |

Un ticket « refused » doit être revu par un superviseur. Si le même problème revient dans les 30 jours après la résolution, le ticket repasse à « reopened » au lieu d'en créer un nouveau.

## Recours du client

Si le client n'est pas satisfait de la réponse du service client, il peut écrire au service consommateurs de CallVerse. Si le désaccord continue après ces deux niveaux de recours internes, il peut saisir gratuitement le Médiateur des communications électroniques. Le conseiller doit donner cette information si le client la demande ou s'il refuse la solution proposée.
