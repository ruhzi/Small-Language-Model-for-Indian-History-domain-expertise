# Dataset taxonomies

Every question in `fully_tagged_dataset.jsonl` carries exactly one tag from each of the two taxonomies below.

## Taxonomy A: time-period coverage (4 buckets)

*When is the dominant historical reasoning anchored?*

Buckets correspond to structural regime changes, not dynasties.

### TP-A: Pre-colonial India (ancient + medieval)

- **Includes**: Harappan, Vedic, Maurya–Gupta; Delhi Sultanate, Mughals; social, cultural and economic systems before British rule.
- **Core reasoning themes**: state formation, civilisational continuity, pre-modern political economy.
- **Excludes**: any question where British rule is causal rather than incidental.
- **Tag if**: the British are absent or only mentioned as a later contrast.

### TP-B: Colonial India (~1757–1947)

- **Includes**: Company rule, 1857 Revolt, national movement, Partition.
- **Core reasoning themes**: colonial extraction, resistance and nationalism, constitutional evolution under empire.
- **Tag if**: British rule is the main driver of the outcomes being analysed.

### TP-C: Early Republic / nation-building (~1947–1977)

- **Includes**: Nehru, the Constitution, Kashmir, planning, Indira Gandhi and the Emergency.
- **Core reasoning themes**: state legitimacy, centralisation vs democracy, ideological foundations.
- This is the dataset's densest bucket by reasoning difficulty.

### TP-D: Contemporary India (~1977–present)

- **Includes**: post-Emergency politics, 1991 reforms, coalition era, Modi, Article 370, identity politics.
- **Core reasoning themes**: institutional change, economic liberalisation, cultural politics.

### Time-period tagging rule (strict)

- Exactly **one** TP tag per question.
- Tag by **where the causal explanation lives**, not by which dates are mentioned.

## Taxonomy B: tonal / question intent (4 buckets)

*What kind of thinking does the question demand?* This is the more important taxonomy for distillation.

### T-A: Explanatory / analytical

"Explain why X happened" or "evaluate X".

- **Includes**: explain, analyse, assess, evaluate.
- **Reasoning style**: structured, cause–effect, multi-factor but not adversarial.
- Lowest ambiguity.

### T-B: Normative / ethical

"Was X right, justified, democratic, fair?"

- **Includes**: moral judgement, rights vs authority, democratic legitimacy.
- **Reasoning style**: value-laden, requires explicit criteria, high chain-of-thought usefulness.

### T-C: Perspective-based / ideological

"How do different sides see X?"

- **Includes**: nationalist vs secular, state vs dissent, competing schools of thought.
- **Reasoning style**: multi-voice, frame-shifting, balance without collapse.

### T-D: Debate / contested

"There is no settled answer."

- **Includes**: Kashmir, Article 370, riots, Emergency legitimacy, identity politics.
- **Reasoning style**: hedging, explicit uncertainty, argument mapping.
- If present, this tag dominates.

### Tonal tagging rules

1. Exactly **one** tonal tag per question.
2. If the question is openly controversial, always tag **T-D**.
3. If several buckets fit, choose the one with the highest reasoning entropy (the most open-ended reasoning style).
