# How this assistant decides whether a recipe suits a condition

This document explains the rules used by the recommender (configs/health_rules.yaml). Values are **per serving** as reported by Food.com and are approximations — real portions and products vary.

| Condition | A recipe is flagged when a serving has… | Basis |
|---|---|---|
| Type 2 diabetes | more than 60 g carbohydrate, or more than 27 g total sugars | ADA carb counting (45–60 g per meal); UK front-of-pack "high sugar" per portion |
| Hypertension | more than 500 mg sodium, or more than 6 g saturated fat | ACC/AHA 2017 ideal limit 1500 mg/day ÷ 3 meals; UK front-of-pack "high saturates" per portion; DASH |
| Hyperuricemia / gout | more than 27 g total sugars, or any high-purine food, alcohol or corn syrup | ACR 2020 gout guideline |

- **Allergies** are a hard rule: recipes whose ingredient list, name or cooking steps mention one of the user's allergens are never recommended.
- Recipes that pass these rules are "compliant"; the assistant shows compliant recipes first and labels any exception clearly.
- These rules support, but do not replace, advice from a doctor or registered dietitian — especially for medication, insulin dosing, kidney disease or pregnancy.
