"""Autoreason-style experiment tournament for autoresearch V2 trainers.

Outer-loop controller: every proposed change to train_v2.py / train_dino_v2.py
competes A (do-nothing incumbent) / B (patch) / AB (synthesis) with blind
rule-based judging; only the winner consumes GPU. Promotion is gated by
objective retrieval eval, not judge ranking.

V2-only: the loop must never write into V1 files (results.tsv, train.py,
train_final.py, train_dino.py, prepare.py). Target adapters enforce this.
"""
