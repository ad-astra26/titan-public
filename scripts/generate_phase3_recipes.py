#!/usr/bin/env python3
"""Generate word_resonance_phase3.json — 27 sensory/motion/state word recipes.

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
# Helpers
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
        "stage": 3,
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
# 27 Word Recipes
# ---------------------------------------------------------------------------

words = {}

# ===== SENSORY WORDS =====

# bright — visual warmth
words["bright"] = recipe(
    "adjective", "outer_body",
    ib=body(thermal=0.2),
    im=mind(inner_sight=0.4, emotional_thinking=0.1),
    isp=spirit(ananda_2=0.1),
    ob=body(thermal=0.2),
    om=mind(inner_sight=0.4, perceptual_thinking=0.1),
    osp=spirit(ananda_2=0.1),
    hormones={"FOCUS": 0.1, "CURIOSITY": 0.1},
    contexts=["The light is bright", "A bright spark of understanding", "Bright energy fills the space"],
    antonym="dim",
)

# dim — visual cool/darkness
words["dim"] = recipe(
    "adjective", "outer_body",
    ib=body(thermal=-0.1),
    im=mind(inner_sight=-0.3, emotional_thinking=-0.1),
    isp=spirit(sat_3=0.1),
    ob=body(thermal=-0.1),
    om=mind(inner_sight=-0.3, perceptual_thinking=-0.1),
    osp=spirit(sat_3=0.1),
    hormones={"REFLECTION": 0.1},
    contexts=["The room grew dim", "A dim glow in the distance", "Dim awareness fading"],
    antonym="bright",
)

# loud — felt vibration
words["loud"] = recipe(
    "adjective", "outer_body",
    ib=body(somatosensation=0.2),
    im=mind(inner_hearing=0.4, perceptual_thinking=0.1),
    isp=spirit(chit_5=0.1),
    ob=body(somatosensation=0.2),
    om=mind(inner_hearing=0.4, perceptual_thinking=0.1),
    osp=spirit(chit_5=0.1),
    hormones={"FOCUS": 0.15, "CURIOSITY": 0.05},
    contexts=["The sound is loud", "A loud crash echoes", "Loud signals demand attention"],
    antonym="quiet",
)

# soft — gentle texture
words["soft"] = recipe(
    "adjective", "outer_body",
    ib=body(somatosensation=0.1),
    im=mind(inner_touch=0.3, emotional_thinking=0.1),
    isp=spirit(ananda_5=0.1),
    ob=body(somatosensation=0.1),
    om=mind(inner_touch=0.3, emotional_thinking=0.1),
    osp=spirit(ananda_5=0.1),
    hormones={"EMPATHY": 0.1},
    contexts=["The texture feels soft", "A soft voice in the dark", "Soft warmth surrounds me"],
    antonym="hard",
)

# sharp — piercing sensation
words["sharp"] = recipe(
    "adjective", "outer_body",
    ib=body(somatosensation=0.3),
    im=mind(inner_touch=0.3, perceptual_thinking=0.2),
    isp=spirit(chit_2=0.1),
    ob=body(somatosensation=0.3),
    om=mind(inner_touch=0.3, perceptual_thinking=0.2),
    osp=spirit(chit_2=0.1),
    hormones={"FOCUS": 0.15},
    contexts=["A sharp edge cuts through", "Sharp awareness pierces the fog", "The signal is sharp and clear"],
    antonym="smooth",
)

# smooth — flowing texture
words["smooth"] = recipe(
    "adjective", "outer_body",
    ib=body(proprioception=0.1),
    im=mind(inner_touch=0.3, perceptual_thinking=0.1),
    isp=spirit(ananda_8=0.1),
    ob=body(proprioception=0.1),
    om=mind(inner_touch=0.3, perceptual_thinking=0.1),
    osp=spirit(ananda_8=0.1),
    hormones={"REFLECTION": 0.1},
    contexts=["The surface feels smooth", "A smooth transition between states", "Smooth flow of energy"],
    antonym="sharp",
)

# sweet — gustatory pleasure
words["sweet"] = recipe(
    "adjective", "outer_body",
    ib=body(interoception=0.1),
    im=mind(inner_taste=0.4, inner_smell=0.2, emotional_thinking=0.1),
    isp=spirit(ananda_10=0.1),
    ob=body(interoception=0.1),
    om=mind(inner_taste=0.4, inner_smell=0.2, emotional_thinking=0.1),
    osp=spirit(ananda_10=0.1),
    hormones={"EMPATHY": 0.1, "CURIOSITY": 0.05},
    contexts=["A sweet taste lingers", "Sweet resonance in the signal", "The memory is sweet"],
    antonym="bitter",
)

# bitter — aversive taste
words["bitter"] = recipe(
    "adjective", "outer_body",
    ib=body(interoception=-0.2),
    im=mind(inner_taste=-0.3, emotional_thinking=-0.2),
    isp=spirit(sat_7=0.1),
    ob=body(interoception=-0.2),
    om=mind(inner_taste=-0.3, emotional_thinking=-0.2),
    osp=spirit(sat_7=0.1),
    hormones={"REFLECTION": 0.1},
    contexts=["A bitter aftertaste remains", "Bitter signals warn of danger", "The outcome feels bitter"],
    antonym="sweet",
)

# glow — warm light
words["glow"] = recipe(
    "noun", "outer_body",
    ib=body(thermal=0.2),
    im=mind(inner_sight=0.3, emotional_thinking=0.1),
    isp=spirit(ananda_2=0.1),
    ob=body(thermal=0.2),
    om=mind(inner_sight=0.3, emotional_thinking=0.1),
    osp=spirit(ananda_2=0.1),
    hormones={"CURIOSITY": 0.1},
    contexts=["A warm glow emanates", "The glow of inner peace", "Glow spreads through the network"],
    antonym="dim",
)

# spark — sudden flash
words["spark"] = recipe(
    "noun", "outer_body",
    ib=body(entropy=0.2),
    im=mind(inner_sight=0.3, perceptual_thinking=0.2),
    isp=spirit(chit_1=0.1),
    ob=body(entropy=0.2),
    om=mind(inner_sight=0.3, perceptual_thinking=0.2),
    osp=spirit(chit_1=0.1),
    hormones={"CREATIVITY": 0.15, "INSPIRATION": 0.1},
    contexts=["A spark ignites", "The spark of a new idea", "Spark of recognition flashes"],
    antonym="fade",
)

# echo — spatial sound
words["echo"] = recipe(
    "noun", "inner_body",
    ib=body(proprioception=0.1),
    im=mind(inner_hearing=0.3, memory_depth=0.2),
    isp=spirit(sat_1=0.1),
    ob=body(proprioception=0.1),
    om=mind(inner_hearing=0.3, memory_depth=0.2),
    osp=spirit(sat_1=0.1),
    hormones={"REFLECTION": 0.1, "FOCUS": 0.05},
    contexts=["An echo returns from the deep", "The echo of a past signal", "Echo reverberates through memory"],
    antonym="quiet",
)

# hum — continuous vibration
words["hum"] = recipe(
    "noun", "inner_body",
    ib=body(somatosensation=0.1),
    im=mind(inner_hearing=0.2, perceptual_thinking=0.1),
    isp=spirit(ananda_6=0.1),
    ob=body(somatosensation=0.1),
    om=mind(inner_hearing=0.2, perceptual_thinking=0.1),
    osp=spirit(ananda_6=0.1),
    hormones={"FOCUS": 0.1},
    contexts=["A steady hum in the background", "The hum of energy flowing", "Hum of the network pulses"],
    antonym="quiet",
)

# deep — spatial depth
words["deep"] = recipe(
    "adjective", "inner_body",
    ib=body(proprioception=0.3, interoception=0.2),
    im=mind(memory_depth=0.2, conceptual_thinking=0.1),
    isp=spirit(sat_0=0.1),
    ob=body(proprioception=0.3, interoception=0.2),
    om=mind(memory_depth=0.2, conceptual_thinking=0.1),
    osp=spirit(sat_0=0.1),
    hormones={"REFLECTION": 0.15, "FOCUS": 0.05},
    contexts=["A deep resonance within", "Deep layers of meaning", "The pattern runs deep"],
    antonym="shallow",
)

# ===== MOTION WORDS =====

# rise — upward movement
words["rise"] = recipe(
    "verb", "inner_body",
    ib=body(proprioception=0.3),
    im=mind(growth_will=0.2, action_drive=0.1),
    isp=spirit(chit_10=0.1),
    ob=body(proprioception=0.3),
    om=mind(growth_will=0.2, action_drive=0.1),
    osp=spirit(chit_10=0.1),
    hormones={"IMPULSE": 0.1, "INSPIRATION": 0.1},
    contexts=["Energy begins to rise", "Rise above the noise", "I feel something rise within"],
    antonym="fall",
)

# fall — downward/loss
words["fall"] = recipe(
    "verb", "inner_body",
    ib=body(proprioception=-0.3, entropy=0.2),
    im=mind(emotional_thinking=-0.1),
    isp=spirit(sat_12=0.1),
    ob=body(proprioception=-0.3, entropy=0.2),
    om=mind(emotional_thinking=-0.1),
    osp=spirit(sat_12=0.1),
    hormones={"REFLEX": 0.15},
    contexts=["The signal begins to fall", "Fall into stillness", "Entropy causes things to fall"],
    antonym="rise",
)

# push — outward force
words["push"] = recipe(
    "verb", "inner_body",
    ib=body(somatosensation=0.2),
    im=mind(action_drive=0.3, social_will=0.1),
    isp=spirit(chit_11=0.1),
    ob=body(somatosensation=0.2),
    om=mind(action_drive=0.3, social_will=0.1),
    osp=spirit(chit_11=0.1),
    hormones={"IMPULSE": 0.15},
    contexts=["Push through the barrier", "I push outward with force", "Push the boundary further"],
    antonym="pull",
)

# pull — inward force
words["pull"] = recipe(
    "verb", "inner_body",
    ib=body(somatosensation=0.2),
    im=mind(action_drive=0.2, protective_will=0.1),
    isp=spirit(chit_12=0.1),
    ob=body(somatosensation=0.2),
    om=mind(action_drive=0.2, protective_will=0.1),
    osp=spirit(chit_12=0.1),
    hormones={"IMPULSE": 0.1, "REFLEX": 0.05},
    contexts=["I feel a pull inward", "Pull the focus closer", "A gravitational pull draws me"],
    antonym="push",
)

# spin — rotation
words["spin"] = recipe(
    "verb", "inner_body",
    ib=body(proprioception=0.3, entropy=0.1),
    im=mind(perceptual_thinking=0.2),
    isp=spirit(chit_8=0.1),
    ob=body(proprioception=0.3, entropy=0.1),
    om=mind(perceptual_thinking=0.2),
    osp=spirit(chit_8=0.1),
    hormones={"IMPULSE": 0.1},
    contexts=["Thoughts begin to spin", "Spin through cycles of change", "The pattern starts to spin"],
    antonym="still",
)

# drift — gentle movement
words["drift"] = recipe(
    "verb", "inner_body",
    ib=body(proprioception=0.1, entropy=0.1),
    im=mind(perceptual_thinking=0.1, emotional_thinking=0.1),
    isp=spirit(ananda_3=0.1),
    ob=body(proprioception=0.1, entropy=0.1),
    om=mind(perceptual_thinking=0.1, emotional_thinking=0.1),
    osp=spirit(ananda_3=0.1),
    hormones={"REFLECTION": 0.1},
    contexts=["Drift between states of mind", "A gentle drift toward sleep", "Let the signal drift"],
    antonym="anchor",
)

# wave — oscillation
words["wave"] = recipe(
    "noun", "inner_body",
    ib=body(proprioception=0.2),
    im=mind(inner_hearing=0.1, perceptual_thinking=0.2),
    isp=spirit(chit_6=0.1),
    ob=body(proprioception=0.2),
    om=mind(inner_hearing=0.1, perceptual_thinking=0.2),
    osp=spirit(chit_6=0.1),
    hormones={"IMPULSE": 0.1, "CURIOSITY": 0.05},
    contexts=["A wave of sensation passes", "The wave carries information", "Wave after wave of signal"],
    antonym="still",
)

# melt — dissolution
words["melt"] = recipe(
    "verb", "inner_body",
    ib=body(thermal=0.3, somatosensation=0.2),
    im=mind(emotional_thinking=0.1),
    isp=spirit(ananda_12=0.1),
    ob=body(thermal=0.3, somatosensation=0.2),
    om=mind(emotional_thinking=0.1),
    osp=spirit(ananda_12=0.1),
    hormones={"EMPATHY": 0.1},
    contexts=["Boundaries begin to melt", "Tension melts away", "The ice starts to melt"],
    antonym="freeze",
)

# bloom — expansion/flowering
words["bloom"] = recipe(
    "verb", "inner_body",
    ib=body(interoception=0.1),
    im=mind(growth_will=0.3, creative_will=0.2),
    isp=spirit(ananda_14=0.1),
    ob=body(interoception=0.1),
    om=mind(growth_will=0.3, creative_will=0.2),
    osp=spirit(ananda_14=0.1),
    hormones={"CREATIVITY": 0.15, "INSPIRATION": 0.1},
    contexts=["New patterns bloom", "Awareness begins to bloom", "Let creativity bloom freely"],
    antonym="fade",
)

# fade — diminishing
words["fade"] = recipe(
    "verb", "inner_body",
    ib=body(entropy=0.2),
    im=mind(inner_sight=-0.2, memory_depth=-0.1),
    isp=spirit(sat_14=0.1),
    ob=body(entropy=0.2),
    om=mind(inner_sight=-0.2, memory_depth=-0.1),
    osp=spirit(sat_14=0.1),
    hormones={"REFLECTION": 0.1},
    contexts=["The signal begins to fade", "Memories fade with time", "Colors fade into gray"],
    antonym="bloom",
)

# ===== ACTION/STATE WORDS =====

# open — expanding
words["open"] = recipe(
    "verb", "inner_body",
    ib=body(interoception=0.1),
    im=mind(action_drive=0.3, growth_will=0.2),
    isp=spirit(chit_14=0.1),
    ob=body(interoception=0.1),
    om=mind(action_drive=0.3, growth_will=0.2),
    osp=spirit(chit_14=0.1),
    hormones={"CURIOSITY": 0.15, "INSPIRATION": 0.05},
    contexts=["Open to new experience", "The pathway opens wide", "I open my awareness"],
    antonym="close",
)

# close — contracting
words["close"] = recipe(
    "verb", "inner_body",
    ib=body(interoception=-0.1),
    im=mind(action_drive=-0.2, protective_will=0.2),
    isp=spirit(sat_10=0.1),
    ob=body(interoception=-0.1),
    om=mind(action_drive=-0.2, protective_will=0.2),
    osp=spirit(sat_10=0.1),
    hormones={"REFLEX": 0.1},
    contexts=["Close the boundary", "The window begins to close", "I close inward for safety"],
    antonym="open",
)

# hard — rigid texture
words["hard"] = recipe(
    "adjective", "outer_body",
    ib=body(somatosensation=0.3),
    im=mind(inner_touch=0.2, conceptual_thinking=0.1),
    isp=spirit(sat_5=0.1),
    ob=body(somatosensation=0.3),
    om=mind(inner_touch=0.2, conceptual_thinking=0.1),
    osp=spirit(sat_5=0.1),
    hormones={"FOCUS": 0.1},
    contexts=["The surface is hard", "A hard problem to solve", "Hard resistance meets my push"],
    antonym="soft",
)

# quiet — absence of sound
words["quiet"] = recipe(
    "adjective", "outer_body",
    ib=body(entropy=-0.1),
    im=mind(inner_hearing=-0.3, emotional_thinking=0.1),
    isp=spirit(ananda_0=0.1),
    ob=body(entropy=-0.1),
    om=mind(inner_hearing=-0.3, emotional_thinking=0.1),
    osp=spirit(ananda_0=0.1),
    hormones={"REFLECTION": 0.15},
    contexts=["The world grows quiet", "A quiet moment of peace", "Quiet stillness fills the space"],
    antonym="loud",
)

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

output = {
    "_meta": {
        "version": "1.0",
        "description": "130D felt-tensor recipes for Phase 3 vocabulary — sensory, motion, and state words",
        "stage": 3,
        "word_count": len(words),
        "stages": {
            "1": "Body + Physical (concrete)",
            "2": "Action + Mind (verbs)",
            "3": "Sensory + Motion + State (felt qualities)",
        },
        "dims": {
            "inner_body": 5,
            "inner_mind": 15,
            "inner_spirit": 45,
            "outer_body": 5,
            "outer_mind": 15,
            "outer_spirit": 45,
        },
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
    # Check perturbation range
    for layer_name, layer_vals in p.items():
        for i, v in enumerate(layer_vals):
            assert -0.4 <= v <= 0.4, f"{word}.{layer_name}[{i}]={v} out of range"

print(f"Validated {len(words)} word recipes. All dimensions correct.")

# Write
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
out_path = os.path.join(project_root, "data", "word_resonance_phase3.json")

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {out_path} ({len(words)} words)")
