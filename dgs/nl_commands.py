"""
Plain-English commands for the 4D viewer: classic NLP, no large language model.

In plain words
--------------
"show the z slice at 0.3 nm at t = 2 fs" has two parts to understand:

1. **Intent:** what kind of view the person wants (slice, projection, probe, ...). This is
   *text classification*. Each phrase is turned into a bag of words (unigrams and bigrams,
   numbers replaced by <num>), and a **multinomial Naive Bayes** classifier picks the most
   likely intent:

       log P(intent | words) = log P(intent) + sum_w count(w) log P(w | intent) + const,
       P(w | intent) = (count(w, intent) + 1) / (total words in intent + vocabulary size)

   The +1 is Laplace smoothing, so an unseen word doesn't zero out a class.
2. **Slots:** the numbers and names the view needs (axis, position, time, point, level,
   max/sum). This is *slot filling*, done with regular expressions.

``parse`` returns a ``Command`` with the intent, the slots, and a confidence (the softmax of
the class scores). Low confidence means "didn't understand", and nothing is guessed.
``execute`` runs a Command against a ``VolumeSeries`` and returns the arrays to draw.

Sources (IEEE style)
--------------------
[1] D. Jurafsky and J. H. Martin, Speech and Language Processing, 3rd ed. draft, 2024,
    ch. 4 (Naive Bayes, Laplace smoothing) and ch. 15 (intents and slot filling).
[2] C. D. Manning, P. Raghavan, and H. Schutze, Introduction to Information Retrieval.
    Cambridge Univ. Press, 2008, ch. 13 (multinomial Naive Bayes).
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

NUM = r"-?\d*\.?\d+"

TRAINING = {
    "slice": [
        "show the z slice at 0.5", "slice along x at 0.3 nm", "cut the volume at y = 0.2",
        "plane at z = 0.7 at t = 1 fs", "show the xy plane", "cross section through the middle",
        "give me a slice perpendicular to z", "cut through x at 0.4 at time 2",
        "show a planar cut at y 0.6", "slice it", "show the yz plane at x = 0.25",
        "cross-section at z = 0.1 nm",
    ],
    "projection": [
        "max projection along z", "maximum intensity projection along y",
        "project onto the xy plane", "sum along x", "collapse the z axis",
        "integrate along y at t = 3 fs", "show the mip along x", "average along z",
        "projection at time 1", "flatten along y with the max", "total along z",
        "show the column sum along x",
    ],
    "probe": [
        "probe the point (0.2, 0.5, 0.5)", "time series at x = 0.3 y = 0.5 z = 0.5",
        "how does the middle voxel change over time", "plot the value at (0.5, 0.5, 0.5) versus time",
        "track one voxel through time", "show the signal at a point", "probe the middle",
        "trace of the density at (0.25, 0.5, 0.5)", "what happens at this point over time",
        "point history at x 0.7 y 0.5 z 0.5", "follow the point (0.1, 0.1, 0.1)",
    ],
    "isosurface": [
        "show the isosurface at 50 percent", "3d blob above half max", "voxels above 30%",
        "render the 3d shape at t = 2", "isosurface at 70% at time 1 fs",
        "where is the density high in 3d", "show the cloud in three dimensions",
        "3d view of the high density region", "contour surface at 0.4 of the peak",
        "draw the 3d volume above 60 percent",
    ],
    "animate": [
        "animate the z slice", "play the projection over time", "make a movie of the isosurface",
        "animate along x", "show it evolving in time", "loop through all the time steps",
        "movie of the max projection along z", "animate the 3d blob", "play the slice at y = 0.5",
        "run the animation", "animate the projection along y", "animate the slice along z over time",
        "play the movie along x", "animate the z slice at z = 0.3", "animate the x slice at x = 0.6 nm",
        "animate the slice at 0.5 over all frames",
    ],
    "peak": [
        "where is the maximum", "when is the peak", "find the global max",
        "what is the highest value", "location of the peak density", "biggest value and when",
        "find the brightest voxel", "when and where does it peak",
    ],
    "center": [
        "track the center of mass", "how does the centroid move", "center of mass over time",
        "where is the average position", "plot the mean position versus time",
        "follow the centroid", "expected position over time", "does the center slosh",
    ],
}


def tokenize(text):
    """Lowercase word and number tokens, numbers mapped to <num>, plus bigrams."""
    raw = re.findall(r"[a-z]+|\d*\.?\d+", text.lower())
    words = ["<num>" if re.fullmatch(r"\d*\.?\d+", w) else w for w in raw]
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


class NaiveBayesIntent:
    """Multinomial Naive Bayes over bag-of-words features, Laplace smoothing alpha."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = alpha

    def fit(self, examples):
        """examples: dict intent -> list of phrases."""
        if not examples or any(len(v) == 0 for v in examples.values()):
            raise ValueError("need at least one phrase per intent")
        self.classes = sorted(examples)
        self.counts = {c: Counter(tok for p in examples[c] for tok in tokenize(p)) for c in self.classes}
        self.vocab = set().union(*self.counts.values())
        n_docs = sum(len(v) for v in examples.values())
        self.log_prior = {c: math.log(len(examples[c]) / n_docs) for c in self.classes}
        self.total = {c: sum(self.counts[c].values()) for c in self.classes}
        return self

    def scores(self, text):
        toks = [t for t in tokenize(text) if t in self.vocab]
        V = len(self.vocab)
        return {c: self.log_prior[c] + sum(math.log((self.counts[c][t] + self.alpha) / (self.total[c] + self.alpha * V))
                                           for t in toks) for c in self.classes}

    def predict(self, text):
        """(best intent, confidence, all class probabilities)."""
        s = self.scores(text)
        m = max(s.values())
        z = {c: math.exp(v - m) for c, v in s.items()}
        tot = sum(z.values())
        probs = {c: v / tot for c, v in z.items()}
        best = max(probs, key=probs.get)
        return best, probs[best], probs


MODEL = NaiveBayesIntent().fit(TRAINING)


@dataclass
class Command:
    text: str
    intent: str
    confidence: float
    slots: dict = field(default_factory=dict)
    understood: bool = True


def extract_slots(text):
    """Pull axis, position, time, point, level, kind and view out of a phrase."""
    s = text.lower()
    slots = {}
    m = re.search(rf"\(\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\)", s)
    coords = {a: float(v) for a, v in re.findall(rf"\b([xyz])\s*(?:=|is|of)?\s*({NUM})", s)}
    if m:
        slots["point"] = tuple(float(g) for g in m.groups())
    elif len(coords) == 3:
        slots["point"] = (coords["x"], coords["y"], coords["z"])
    m = re.search(rf"\b(?:t|time)\s*(?:=|is|of)?\s*({NUM})|({NUM})\s*(?:fs|femtoseconds?)\b", s)
    if m:
        slots["t"] = float(m.group(1) or m.group(2))
    m = re.search(r"\b(xy|yz|xz|yx|zy|zx)[- ]?plane", s)
    if m:
        slots["axis"] = ({"x", "y", "z"} - set(m.group(1))).pop()
    for pat in (r"along (?:the )?([xyz])\b", r"\b([xyz])[- ]axis", r"perpendicular to ([xyz])\b",
                r"\b([xyz])[- ]slice", r"(?:through|at) ([xyz])\b"):
        m = re.search(pat, s)
        if m and "axis" not in slots:
            slots["axis"] = m.group(1)
    if "point" not in slots and len(coords) >= 1:
        a, v = next(iter(coords.items()))
        slots.setdefault("axis", a)
        if slots["axis"] == a:
            slots["position"] = v
    m = re.search(rf"({NUM})\s*(?:%|percent)", s)
    if m:
        slots["level"] = float(m.group(1)) / 100
    elif "half max" in s or "half the max" in s:
        slots["level"] = 0.5
    else:
        m = re.search(rf"({NUM}) of the (?:peak|max)", s)
        if m:
            slots["level"] = float(m.group(1))
    if re.search(r"\b(sum|total|integrat\w*|column)\b", s):
        slots["kind"] = "sum"
    elif re.search(r"\b(average|mean)\b", s) and "position" not in s:
        slots["kind"] = "mean"
    if re.search(r"\b(iso\w*|blob|voxels?|3d|cloud)\b", s):
        slots["view"] = "isosurface"
    elif re.search(r"\b(projection|project|mip|collapse|sum|max)\b", s):
        slots["view"] = "projection"
    elif re.search(r"\b(slice|plane|cut|section)\b", s):
        slots["view"] = "slice"
    return slots


def parse(text, threshold=0.4):
    """Classify the intent and fill the slots. Below `threshold` confidence: understood=False."""
    if not text or not text.strip():
        raise ValueError("empty command")
    intent, conf, _ = MODEL.predict(text)
    return Command(text, intent, conf, extract_slots(text), understood=conf >= threshold)


def execute(vs, cmd):
    """Run a Command against a VolumeSeries. Returns a dict describing what to draw."""
    if not cmd.understood:
        return {"kind": "unknown", "text": f"Not sure what you meant (best guess '{cmd.intent}', "
                                           f"{cmd.confidence:.0%}). Try e.g. 'z slice at 0.5 at t = 1 fs'."}
    sl = cmd.slots
    t = sl.get("t", float(vs.t[0]))
    axis = sl.get("axis", "z")
    pos = sl.get("position", float(np.median(vs.axis(axis))))
    if cmd.intent == "slice":
        img, hv = vs.slice(axis, pos, t)
        return {"kind": "image", "image": img, "axes": hv, "text": f"{axis} = {pos:g} {vs.units[axis]} slice at t = {t:g} {vs.units['t']}"}
    if cmd.intent == "projection":
        kind = sl.get("kind", "max")
        img, hv = vs.projection(axis, t, kind)
        return {"kind": "image", "image": img, "axes": hv, "text": f"{kind} projection along {axis} at t = {t:g} {vs.units['t']}"}
    if cmd.intent == "probe":
        pt = sl.get("point", tuple(float(np.median(vs.axis(a))) for a in "xyz"))
        return {"kind": "series", "x": vs.t, "y": vs.probe(pt), "text": f"time series at (x, y, z) = {pt}"}
    if cmd.intent == "isosurface":
        lvl = sl.get("level", 0.5)
        pts, vals = vs.isosurface_points(t, lvl)
        return {"kind": "points", "points": pts, "values": vals, "text": f"voxels above {lvl:.0%} of the max at t = {t:g} {vs.units['t']}"}
    if cmd.intent == "animate":
        return {"kind": "animation", "view": sl.get("view", "slice"), "axis": axis, "position": pos,
                "level": sl.get("level", 0.5), "how": sl.get("kind", "max"),
                "text": f"animate the {sl.get('view', 'slice')} ({axis}) over {len(vs.t)} frames"}
    if cmd.intent == "peak":
        tp, xyz, v = vs.peak()
        return {"kind": "answer", "text": f"peak {v:.4g} at t = {tp:.3g} {vs.units['t']}, (x, y, z) = ({xyz[0]:.3g}, {xyz[1]:.3g}, {xyz[2]:.3g})"}
    if cmd.intent == "center":
        return {"kind": "series3", "x": vs.t, "y": vs.center_of_mass(), "text": "center of mass (x, y, z) vs time"}
    raise ValueError(f"unknown intent {cmd.intent!r}")
