"""Directional phrases chosen for monetary-policy meaning, not generic sentiment."""

HAWKISH_PHRASES = {
    "inflation remains elevated": 2.0,
    "inflation is elevated": 2.0,
    "inflationary pressures": 1.5,
    "upside risks to inflation": 2.0,
    "inflation has increased": 2.0,
    "inflation has risen": 2.0,
    "persistent inflation": 2.0,
    "inflation remains above": 1.5,
    "labor market remains tight": 1.5,
    "strong labor market": 1.0,
    "economic activity remains strong": 1.0,
    "restrictive stance": 1.0,
    "raise the target range": 2.5,
    "increase the target range": 2.5,
    "further tightening": 2.0,
    "additional firming": 2.0,
    "higher for longer": 2.0,
    "not appropriate to reduce": 2.0,
    "not yet gained confidence": 1.5,
    "vigilant to inflation risks": 1.5,
}

DOVISH_PHRASES = {
    "inflation has eased": 2.0,
    "inflation has declined": 2.0,
    "inflation has moderated": 2.0,
    "disinflation": 1.5,
    "downside risks to employment": 2.0,
    "downside risks to growth": 2.0,
    "labor market has cooled": 1.5,
    "labor market conditions have eased": 1.5,
    "economic activity has slowed": 1.5,
    "weaker economic activity": 1.0,
    "reduce the target range": 2.5,
    "lower the target range": 2.5,
    "policy easing": 2.0,
    "less restrictive": 1.5,
    "appropriate to reduce": 2.0,
    "gained greater confidence": 1.5,
    "support maximum employment": 1.0,
    "risks are roughly in balance": 0.5,
    "weakened labor market": 1.5,
    "rate cuts": 2.0,
}

HAWKISH_ANCHORS = [
    "Interest rates will need to rise to restrain inflation.",
    "Inflation remains too high and upside inflation risks have increased.",
    "The labor market is tight and further monetary policy tightening is appropriate.",
    "Policy should remain restrictive for longer than previously expected.",
    "The Committee is not yet confident that inflation is moving sustainably to target.",
]

DOVISH_ANCHORS = [
    "Interest rates should be reduced to support employment and economic activity.",
    "Inflation has eased and downside risks to growth have increased.",
    "The labor market has cooled and monetary policy can become less restrictive.",
    "The Committee has gained confidence that inflation is moving sustainably to target.",
    "Policy easing is appropriate because economic activity has slowed.",
]

