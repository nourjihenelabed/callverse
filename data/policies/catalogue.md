# Catalogue CallVerse Telecom

Prix en euros TTC, par mois. Tous les prix de ce document sont aussi utilisés par `scripts/seed_data.py` pour calculer les factures des clients : si vous changez un prix ici, changez-le aussi dans `CATALOGUE` de ce script (un test vérifie que les deux sont identiques).

## Box internet

| Code | Offre | Prix mensuel | Description |
| --- | --- | --- | --- |
| box_fibre | Box Fibre | 34,99 € | Internet très haut débit par la fibre optique, téléphone fixe inclus. |
| box_adsl | Box ADSL | 24,99 € | Internet haut débit par la ligne téléphonique (ADSL/VDSL), téléphone fixe inclus. |

## Mobile

| Code | Offre | Prix mensuel | Description |
| --- | --- | --- | --- |
| forfait_mobile | Forfait Mobile 100 Go | 14,99 € | Appels, SMS et MMS illimités en France, 100 Go d'internet mobile. |
| ligne_supplementaire | Ligne mobile supplémentaire | 9,99 € | Une ligne en plus sur le même compte, avec le même forfait. |

## Options

Les options s'ajoutent à une box internet. Elles sont sans engagement et résiliables à tout moment depuis l'espace client.

| Code | Option | Prix mensuel | Description |
| --- | --- | --- | --- |
| option_securite | Sécurité Internet | 3,99 € | Antivirus et contrôle parental pour 5 appareils. |
| option_sauvegarde | Sauvegarde en ligne | 2,99 € | 100 Go de stockage en ligne pour les photos et documents. |
| option_protection_appareil | Protection des appareils | 4,99 € | Réparation ou remplacement de la box et du décodeur en cas de panne ou de casse. |
| option_assistance_premium | Assistance Premium | 5,99 € | Conseiller technique dédié, 7j/7, avec rappel sous 2 heures. |
| option_tv | TV | 7,99 € | Plus de 150 chaînes de télévision sur le décodeur et l'application. |
| option_cinema | Cinéma & Séries | 9,99 € | Catalogue de films et séries à la demande. |

## Durée d'engagement

| Code | Engagement |
| --- | --- |
| sans_engagement | Sans engagement, résiliable à tout moment. |
| engagement_12_mois | Engagement de 12 mois. |
| engagement_24_mois | Engagement de 24 mois. |

Les conditions de résiliation et les frais éventuels sont décrits dans `contracts.md`.

## Moyens de paiement

| Code | Moyen de paiement |
| --- | --- |
| prelevement | Prélèvement automatique sur le compte bancaire |
| carte_bancaire | Paiement automatique par carte bancaire |
| paiement_en_ligne | Paiement en ligne chaque mois depuis l'espace client |
| cheque | Chèque envoyé par courrier |
