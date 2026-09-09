# Script de démo live — 5 minutes

Calé sur `demo.pptx`. Les slides encadrent la démo ; l'essentiel du temps se passe dans
l'app elle-même. Chronomètre en main, pas au feeling — 5 min passent vite.

---

## 0:00 – 0:35 — Intro (slides 1-2)

*Slide 1 (titre)* : te présenter en une phrase — nom, bootcamp BeCode, track GenAI Developer.

*Slide 2 (le brief)* : "On m'a donné un brief façon second tour d'entretien : construire en
5 jours une app d'optimisation de portefeuille. Trois choses étaient demandées — optimiser,
prévoir, comparer. Je vais vous montrer les trois, plus ce que j'ai ajouté au-delà."

Ne pas lire les bullets à l'écran mot pour mot — la slide est un support visuel, pas un
prompteur.

## 0:35 – 1:05 — Choix techniques (slide 3)

"Quatre couches : yfinance pour la donnée, PyPortfolioOpt pour l'optimisation,
statsmodels pour la prévision — préféré à Kats, qui n'est plus maintenu — et Groq/Ollama
pour la couche IA, avec un vrai fallback local si Groq tombe."

Ne pas s'attarder — une phrase par couche, pas de justification longue ici (ça, c'est pour
les questions après).

## 1:05 – 1:15 — Transition (slide 4)

Juste l'annoncer : "Je passe sur l'app en direct." Ne pas la commenter davantage, elle sert
de repère visuel pendant que tu bascules d'écran.

## 1:15 – 3:30 — DÉMO LIVE dans l'app (2 min 15)

Ordre et budget-temps (serré — respecte-le, ne t'attarde sur aucun onglet) :

1. **Overview → Macro & Risk** (20s) — pointer le taux sans risque en live via FRED, le VIX,
   le spread 10Y-3M. "Le taux sans risque n'est pas codé en dur, il vient de FRED en direct."
2. **Efficient Frontier** (35s) — montrer le nuage de points, la frontière, l'étoile max-Sharpe.
   Ouvrir le tableau des poids optimaux. Mentionner le ratio de diversification en une phrase.
3. **Forecast & Compare** (40s) — LE cœur de la démo. Montrer le tableau Historical /
   Forecast-based / Realized-optimal côte à côte. Dire explicitement : "Realized-optimal,
   c'est de la triche assumée — c'est l'optimum avec les vrais prix futurs, connu seulement
   après coup. Il sert de benchmark, jamais d'objectif atteignable."
4. **Walk-forward validation** (30s) — le box plot multi-fenêtres. "Une seule fenêtre de
   test peut être un coup de chance. Ici c'est répété sur plusieurs fenêtres glissantes."
   Donner le taux de victoire du forecast (ex. 67%) sans détailler le calcul.
5. **AI Analyst / chatbot** (10-20s, SI le temps le permet) — poser une question déjà
   préparée en direct ("pourquoi NVDA est-il autant pondéré ?") pour montrer que la réponse
   est ancrée dans les vrais chiffres calculés, pas inventée.

**Si tu es en retard à 3:00**, saute directement le chatbot et enchaîne sur la slide 5 —
ne rogne jamais sur Forecast & Compare, c'est la pièce centrale du brief.

## 3:30 – 4:10 — Résultats & différenciateurs (slides 5-7)

*Slide 5* : un seul chiffre à retenir à voix haute (le Sharpe ratio, 1.99) — le reste, les
gens le lisent eux-mêmes.

*Slide 6* : reprendre le tableau déjà vu en live, dire la phrase de conclusion : "Le forecast
apporte un edge réel mais modeste — cohérent avec l'hypothèse de marché efficient, pas un
signal miracle." C'est le genre de phrase qui rassure un jury sur ta rigueur.

*Slide 7* : passer vite, une respiration après le rythme de la démo — "Au-delà du brief :
walk-forward, GARCH, RAG, panel macro. Tout est testé, 314 tests, mypy strict."

## 4:10 – 4:45 — Méthodologie IA (slide 8)

C'est ton différenciateur pour le track GenAI Developer — ne le bâcle pas mais reste bref.
"Je n'ai pas juste demandé à une IA d'écrire du code : j'ai construit un vrai garde-fou —
spec précise, contexte centralisé, boucle de tests. Voici 4 vrais bugs, comment ils ont été
trouvés, comment ils sont devenus des règles permanentes." Citer UN exemple en détail
(le CIK SEC EDGAR est le plus parlant — un vrai faux-positif, trouvé en testant en direct).

## 4:45 – 5:00 — Clôture (slide 9)

"Merci — questions ?" Rien d'autre. Ne pas répéter le lien du repo à l'oral, il est affiché.

---

## Filet de sécurité

- **Si le déploiement (Render/Streamlit Cloud) est lent au démarrage** : ouvrir l'app 2-3
  minutes AVANT de commencer à parler, dans un onglet en arrière-plan, pour laisser le temps
  au cold start du tier gratuit.
- **Si une question technique arrive pendant la démo** : noter mentalement, répondre après
  la clôture — ne jamais interrompre le chronomètre des 5 minutes pour y répondre en plein
  milieu.
- **Question probable à préparer** : "Pourquoi pas Riskfolio-Lib / Kats ?" → réponse courte
  déjà dans le README, la relire une fois avant l'oral pour la sortir sans hésiter.
