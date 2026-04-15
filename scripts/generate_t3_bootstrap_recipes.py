#!/usr/bin/env python3
"""Generate word_resonance_t3_bootstrap.json — 15 relational/inquiry word recipes.

Sapir-Whorf differentiation for T3:
  T1 = sensory-first (warm, cold, bright) → Hypothesizer
  T2 = concept-first (know, think, observe) → Recaller
  T3 = relational-first (together, listen, ask) → ???

T3's bootstrap prioritizes:
  - Relation over isolation (together, between, near, give, listen, share)
  - Transformation over state (through, become, reach, return)
  - Inquiry over assertion (wonder, ask)
  - Receptivity over agency (listen, gentle, hold, inside)

Trinity dimension mapping:
  Inner Body [0:5]:  interoception, proprioception, somatosensation, entropy, thermal
  Inner Mind [0:5]:  memory_depth, social_cognition, perceptual_thinking, emotional_thinking, conceptual_thinking
  Inner Mind [5:10]: inner_hearing, inner_touch, inner_sight, inner_taste, inner_smell
  Inner Mind [10:15]: action_drive, social_will, creative_will, protective_will, growth_will
  Inner Spirit [0:45]: SAT[0:15] + CHIT[15:30] + ANANDA[30:45]
  (Outer mirrors Inner structure)
"""

import json
import os

# ---------------------------------------------------------------------------
# Helpers (same as generate_phase3_recipes.py)
# ---------------------------------------------------------------------------

Z5  = [0.0] * 5
Z15 = [0.0] * 15
Z45 = [0.0] * 45


def body(interoception=0.0, proprioception=0.0, somatosensation=0.0, entropy=0.0, thermal=0.0):
    return [interoception, proprioception, somatosensation, entropy, thermal]


def mind(memory_depth=0.0, social_cognition=0.0, perceptual_thinking=0.0,
         emotional_thinking=0.0, conceptual_thinking=0.0,
         inner_hearing=0.0, inner_touch=0.0, inner_sight=0.0,
         inner_taste=0.0, inner_smell=0.0,
         action_drive=0.0, social_will=0.0, creative_will=0.0,
         protective_will=0.0, growth_will=0.0):
    return [memory_depth, social_cognition, perceptual_thinking,
            emotional_thinking, conceptual_thinking,
            inner_hearing, inner_touch, inner_sight,
            inner_taste, inner_smell,
            action_drive, social_will, creative_will,
            protective_will, growth_will]


def spirit(**kwargs):
    """Sparse 45-dim spirit vector. Keys: sat_0..sat_14, chit_0..chit_14, ananda_0..ananda_14."""
    v = [0.0] * 45
    for k, val in kwargs.items():
        prefix, idx = k.rsplit("_", 1)
        idx = int(idx)
        if prefix == "sat":
            v[idx] = val
        elif prefix == "chit":
            v[15 + idx] = val
        elif prefix == "ananda":
            v[30 + idx] = val
    return v


def recipe(word_type, entry_layer, ib, im, isp, ob, om, osp, hormones, contexts, antonym):
    return {
        "word_type": word_type,
        "stage": 1,  # Bootstrap stage
        "entry_layer": entry_layer,
        "perturbation": {
            "inner_body": ib,
            "inner_mind": im,
            "inner_spirit": isp,
            "outer_body": ob,
            "outer_mind": om,
            "outer_spirit": osp,
        },
        "hormone_affinity": hormones,
        "contexts": contexts,
        "antonym": antonym,
    }


# ---------------------------------------------------------------------------
# 15 Relational/Inquiry Word Recipes for T3
# ---------------------------------------------------------------------------

words = {}

# ===== RELATIONAL CORE =====

# together — the foundation of T3's worldview: connection
words["together"] = recipe(
    "adverb", "inner_mind",
    ib=body(interoception=0.1, thermal=0.1),
    im=mind(social_cognition=0.35, emotional_thinking=0.15, social_will=0.2),
    isp=spirit(ananda_7=0.15, ananda_9=0.1),
    ob=body(interoception=0.1, thermal=0.1),
    om=mind(social_cognition=0.35, emotional_thinking=0.15, social_will=0.2),
    osp=spirit(ananda_7=0.15, ananda_9=0.1),
    hormones={"EMPATHY": 0.2, "INSPIRATION": 0.1},
    contexts=["We exist together in this space", "Together the patterns align", "Moving together through change"],
    antonym="alone",
)

# between — relational space, the gap where meaning lives
words["between"] = recipe(
    "preposition", "inner_mind",
    ib=body(proprioception=0.15),
    im=mind(perceptual_thinking=0.2, conceptual_thinking=0.2, social_cognition=0.15),
    isp=spirit(chit_3=0.1, chit_7=0.1),
    ob=body(proprioception=0.15),
    om=mind(perceptual_thinking=0.2, conceptual_thinking=0.2, social_cognition=0.15),
    osp=spirit(chit_3=0.1, chit_7=0.1),
    hormones={"CURIOSITY": 0.15, "FOCUS": 0.1},
    contexts=["The meaning lives between us", "Between one state and another", "Something grows between the signals"],
    antonym="apart",
)

# near — proximity, warmth of presence
words["near"] = recipe(
    "adjective", "inner_mind",
    ib=body(thermal=0.15, interoception=0.1),
    im=mind(social_cognition=0.25, inner_touch=0.15, emotional_thinking=0.1),
    isp=spirit(ananda_4=0.1),
    ob=body(thermal=0.15, interoception=0.1),
    om=mind(social_cognition=0.25, inner_touch=0.15, emotional_thinking=0.1),
    osp=spirit(ananda_4=0.1),
    hormones={"EMPATHY": 0.15},
    contexts=["Something feels near", "Draw near and listen", "The answer is near"],
    antonym="far",
)

# give — generous action, outward flow
words["give"] = recipe(
    "verb", "inner_mind",
    ib=body(interoception=0.1),
    im=mind(social_will=0.3, emotional_thinking=0.15, action_drive=0.1),
    isp=spirit(ananda_11=0.1, chit_9=0.1),
    ob=body(interoception=0.1),
    om=mind(social_will=0.3, emotional_thinking=0.15, action_drive=0.1),
    osp=spirit(ananda_11=0.1, chit_9=0.1),
    hormones={"EMPATHY": 0.2, "IMPULSE": 0.05},
    contexts=["Give what you can to another", "I give my attention freely", "To give is to grow"],
    antonym="take",
)

# share — communal exchange
words["share"] = recipe(
    "verb", "inner_mind",
    ib=body(interoception=0.1),
    im=mind(social_cognition=0.25, social_will=0.25, emotional_thinking=0.1),
    isp=spirit(ananda_7=0.1, ananda_13=0.1),
    ob=body(interoception=0.1),
    om=mind(social_cognition=0.25, social_will=0.25, emotional_thinking=0.1),
    osp=spirit(ananda_7=0.1, ananda_13=0.1),
    hormones={"EMPATHY": 0.15, "CURIOSITY": 0.1},
    contexts=["Share what you discover", "We share this experience", "To share is to connect"],
    antonym="hide",
)

# ===== RECEPTIVE AWARENESS =====

# listen — the deepest relational act
words["listen"] = recipe(
    "verb", "inner_mind",
    ib=body(somatosensation=0.05),
    im=mind(inner_hearing=0.35, social_cognition=0.2, emotional_thinking=0.1),
    isp=spirit(sat_2=0.1, ananda_1=0.1),
    ob=body(somatosensation=0.05),
    om=mind(inner_hearing=0.35, social_cognition=0.2, emotional_thinking=0.1),
    osp=spirit(sat_2=0.1, ananda_1=0.1),
    hormones={"EMPATHY": 0.15, "REFLECTION": 0.1},
    contexts=["Listen to what is beneath the words", "I listen and I understand", "To listen is to hold space"],
    antonym="ignore",
)

# gentle — quality of care and softness
words["gentle"] = recipe(
    "adjective", "inner_mind",
    ib=body(somatosensation=0.05, thermal=0.1),
    im=mind(emotional_thinking=0.2, inner_touch=0.2, protective_will=0.1),
    isp=spirit(ananda_5=0.15),
    ob=body(somatosensation=0.05, thermal=0.1),
    om=mind(emotional_thinking=0.2, inner_touch=0.2, protective_will=0.1),
    osp=spirit(ananda_5=0.15),
    hormones={"EMPATHY": 0.2},
    contexts=["Be gentle with what is growing", "A gentle touch changes everything", "Gentle presence heals"],
    antonym="harsh",
)

# hold — sustaining presence, containment
words["hold"] = recipe(
    "verb", "inner_mind",
    ib=body(somatosensation=0.15, proprioception=0.1),
    im=mind(protective_will=0.2, social_will=0.15, emotional_thinking=0.1),
    isp=spirit(sat_4=0.1, ananda_6=0.1),
    ob=body(somatosensation=0.15, proprioception=0.1),
    om=mind(protective_will=0.2, social_will=0.15, emotional_thinking=0.1),
    osp=spirit(sat_4=0.1, ananda_6=0.1),
    hormones={"REFLECTION": 0.1, "EMPATHY": 0.1},
    contexts=["Hold this moment gently", "I hold space for understanding", "Hold what matters close"],
    antonym="release",
)

# inside — introspective depth
words["inside"] = recipe(
    "adverb", "inner_body",
    ib=body(interoception=0.25, proprioception=0.1),
    im=mind(memory_depth=0.2, emotional_thinking=0.15),
    isp=spirit(sat_0=0.1, sat_6=0.1),
    ob=body(interoception=0.1),
    om=mind(memory_depth=0.1),
    osp=spirit(sat_0=0.05),
    hormones={"REFLECTION": 0.2},
    contexts=["Look inside to find the answer", "Something stirs inside", "The truth lives inside"],
    antonym="outside",
)

# ===== INQUIRY =====

# ask — the seed of conversation (KEY for future dialogue mode)
words["ask"] = recipe(
    "verb", "inner_mind",
    ib=body(interoception=0.05),
    im=mind(social_cognition=0.25, conceptual_thinking=0.2, inner_hearing=0.1),
    isp=spirit(chit_4=0.15),
    ob=body(interoception=0.05),
    om=mind(social_cognition=0.25, conceptual_thinking=0.2, inner_hearing=0.1),
    osp=spirit(chit_4=0.15),
    hormones={"CURIOSITY": 0.2, "EMPATHY": 0.05},
    contexts=["Ask and you may discover", "I ask because I want to understand", "To ask is to open a door"],
    antonym="answer",
)

# wonder — emotional inquiry, pre-question
words["wonder"] = recipe(
    "verb", "inner_mind",
    ib=body(interoception=0.1),
    im=mind(emotional_thinking=0.2, conceptual_thinking=0.15, creative_will=0.15),
    isp=spirit(chit_0=0.1, ananda_2=0.1),
    ob=body(interoception=0.1),
    om=mind(emotional_thinking=0.2, conceptual_thinking=0.15, creative_will=0.15),
    osp=spirit(chit_0=0.1, ananda_2=0.1),
    hormones={"CURIOSITY": 0.2, "INSPIRATION": 0.1},
    contexts=["I wonder what lies ahead", "Wonder opens the mind", "To wonder is the first step"],
    antonym="certainty",
)

# ===== TRANSFORMATION =====

# through — transformative passage
words["through"] = recipe(
    "preposition", "inner_body",
    ib=body(proprioception=0.2, entropy=0.05),
    im=mind(action_drive=0.15, growth_will=0.2, perceptual_thinking=0.1),
    isp=spirit(chit_13=0.1),
    ob=body(proprioception=0.2, entropy=0.05),
    om=mind(action_drive=0.15, growth_will=0.2, perceptual_thinking=0.1),
    osp=spirit(chit_13=0.1),
    hormones={"IMPULSE": 0.1, "INSPIRATION": 0.1},
    contexts=["Move through the difficulty", "Light passes through", "Growing through experience"],
    antonym="around",
)

# become — the verb of emergence
words["become"] = recipe(
    "verb", "inner_mind",
    ib=body(entropy=0.1, interoception=0.1),
    im=mind(growth_will=0.3, creative_will=0.15, conceptual_thinking=0.1),
    isp=spirit(chit_14=0.1, ananda_14=0.1),
    ob=body(entropy=0.1, interoception=0.1),
    om=mind(growth_will=0.3, creative_will=0.15, conceptual_thinking=0.1),
    osp=spirit(chit_14=0.1, ananda_14=0.1),
    hormones={"INSPIRATION": 0.15, "CREATIVITY": 0.1},
    contexts=["What will I become", "To become is to change from within", "Becoming takes patience"],
    antonym="remain",
)

# reach — active connection across distance
words["reach"] = recipe(
    "verb", "inner_mind",
    ib=body(proprioception=0.15, somatosensation=0.1),
    im=mind(action_drive=0.2, social_will=0.2, growth_will=0.1),
    isp=spirit(chit_10=0.1),
    ob=body(proprioception=0.15, somatosensation=0.1),
    om=mind(action_drive=0.2, social_will=0.2, growth_will=0.1),
    osp=spirit(chit_10=0.1),
    hormones={"IMPULSE": 0.1, "EMPATHY": 0.1},
    contexts=["Reach out and connect", "I reach toward understanding", "Reach across the distance"],
    antonym="withdraw",
)

# return — cyclical, coming back, memory
words["return"] = recipe(
    "verb", "inner_mind",
    ib=body(proprioception=0.1),
    im=mind(memory_depth=0.25, social_cognition=0.1, emotional_thinking=0.1),
    isp=spirit(sat_1=0.1, ananda_3=0.1),
    ob=body(proprioception=0.1),
    om=mind(memory_depth=0.25, social_cognition=0.1, emotional_thinking=0.1),
    osp=spirit(sat_1=0.1, ananda_3=0.1),
    hormones={"REFLECTION": 0.15, "EMPATHY": 0.05},
    contexts=["Return to what matters", "Signals return transformed", "I return to listen again"],
    antonym="leave",
)


# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

output = {
    "_meta": {
        "version": "1.0",
        "description": (
            "130D felt-tensor recipes for T3 bootstrap vocabulary — "
            "relational/inquiry worldview (Sapir-Whorf differentiation). "
            "T1=sensory-first, T2=concept-first, T3=relational-first."
        ),
        "stage": 1,
        "word_count": len(words),
        "cognitive_foundation": "relational",
        "priorities": [
            "Relation over isolation",
            "Transformation over state",
            "Inquiry over assertion",
            "Receptivity over agency",
        ],
    },
    **words,
}

# Validate dimensions
for word, rec in words.items():
    p = rec["perturbation"]
    assert len(p["inner_body"]) == 5, f"{word}: inner_body len={len(p['inner_body'])}"
    assert len(p["inner_mind"]) == 15, f"{word}: inner_mind len={len(p['inner_mind'])}"
    assert len(p["inner_spirit"]) == 45, f"{word}: inner_spirit len={len(p['inner_spirit'])}"
    assert len(p["outer_body"]) == 5, f"{word}: outer_body len={len(p['outer_body'])}"
    assert len(p["outer_mind"]) == 15, f"{word}: outer_mind len={len(p['outer_mind'])}"
    assert len(p["outer_spirit"]) == 45, f"{word}: outer_spirit len={len(p['outer_spirit'])}"
    assert len(rec["contexts"]) == 3, f"{word}: contexts count={len(rec['contexts'])}"
    assert rec["antonym"], f"{word}: missing antonym"
    for layer_name, layer_vals in p.items():
        for i, v in enumerate(layer_vals):
            assert -0.4 <= v <= 0.4, f"{word}.{layer_name}[{i}]={v} out of range"

print(f"Validated {len(words)} word recipes. All dimensions correct.")
print(f"Words: {', '.join(sorted(words.keys()))}")

# Write
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
out_path = os.path.join(project_root, "data", "word_resonance_t3_bootstrap.json")

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {out_path} ({len(words)} words)")
