#!/usr/bin/env python3
"""Check every number the paper prints against the committed result files.

    python3 reproduce.py            # reads results/*.json only: CPU, seconds, no data download
    python3 reproduce.py --rerun    # first re-runs steps 1, 3a and 4 from cached inputs, then checks

Each check recomputes one printed number from `results/*.json` (or from the per-session rows inside them)
and passes when the recomputed value rounds to the digits the paper prints. Counts must match exactly.
The exit status is 0 only if every check passes.

`--rerun` needs Cinà's repository cloned into $RESCORING_DRIFT_DATA/cina (see README). It re-executes
`src/step1_sessions.py`, `src/step1_headroom.py`, `src/step3a_gate.py` (T15 and T12, with and without the
training-text overlap) and `src/step4_rescoring.py` (from the pod outputs committed in `derived/step4/`),
reports which result files changed byte for byte, and then runs the same checks on the new files.
Steps 2 and 3b and the step 4 decode need the neural data, a GPU or the model weights; the README lists
their commands, cost and time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load(name):
    return json.loads((RESULTS / name).read_text())


def get(obj, dotted):
    for key in dotted.split("."):
        obj = obj[key]
    return obj


class Checks:
    def __init__(self):
        self.rows, self.failed = [], 0
        self.section = ""

    def head(self, section):
        self.section = section

    def add(self, claim, observed, printed, decimals=None):
        """`printed` is the value as the paper prints it; `decimals` its number of decimal places.
        Integers must match exactly; anything else must round to the printed digits."""
        if decimals is None:
            ok = observed == printed
            obs = str(observed)
        else:
            ok = abs(float(observed) - float(printed)) <= 0.5 * 10 ** -decimals + 1e-9
            obs = f"{float(observed):.{decimals + 2}f}"
        self.failed += not ok
        self.rows.append((self.section, claim, str(printed), obs, ok))

    def report(self):
        w = max(len(r[1]) for r in self.rows)
        last = None
        for section, claim, printed, obs, ok in self.rows:
            if section != last:
                print(f"\n{section}")
                last = section
            print(f"  {claim.ljust(w)}  paper {printed:>9}  recomputed {obs:>11}  {'PASS' if ok else 'FAIL'}")
        n = len(self.rows)
        print(f"\n{n - self.failed}/{n} checks passed")
        return self.failed == 0


def check_data_and_splits(c):
    c.head("§2.1 Data and sessions")
    m = load("step1_sessions.json")
    val = load("val_sessions.json")
    c.add("T15 sessions with validation data", len({r["session"] for r in val}), 41)
    c.add("T15 validation trials", m["mapping"]["validation_trials"], 1426)
    c.add("T15 distinct sentences", m["mapping"]["distinct_sentences"], 1397)
    c.add("T15 sentences in several sessions", m["mapping"]["sentences_in_several_sessions"], 17)
    c.add("T15 trials of those sentences", m["mapping"]["trials_whose_sentence_repeats"], 45)
    c.add("T15 lists with a unique session", m["mapping"]["cached_lists_with_unique_session"], 1380)
    t = load("step1_sessions_t12.json")
    t12 = load("t12_sessions.json")
    c.add("T12 cached lists", t["mapping"]["cached_lists"], 877)
    c.add("T12 test-split trials", t["mapping"]["validation_trials"], 880)
    c.add("T12 sessions", len({r["session"] for r in t12}), 24)
    c.add("T12 lists with a unique session", t["mapping"]["cached_lists_with_unique_session"], 874)
    c.add("T15 fit sessions", len(m["temporal"]["early_sessions"]), 21)
    c.add("T15 fit sentences", m["temporal"]["early_sentences"], 729)
    c.add("T15 test sessions", len(m["temporal"]["late_sessions"]), 20)
    c.add("T15 test sentences", m["temporal"]["late_sentences"], 651)
    c.add("T12 fit sessions", len(t["temporal"]["early_sessions"]), 14)
    c.add("T12 fit sentences", t["temporal"]["early_sentences"], 474)
    c.add("T12 test sessions", len(t["temporal"]["late_sessions"]), 10)
    c.add("T12 test sentences", t["temporal"]["late_sentences"], 400)
    c.add("random test sentences with a session", load("step1_headroom.json")["test_sentences"], 966)

    c.head("§2.2, §2.5, §2.6 Methods")
    early = load("step2_per_early.json")
    c.add("early model: sessions with training data", early["trained_sessions"], 22)
    c.add("early model: trained sessions with validation data",
          sum(s["trained_on"] for s in early["sessions"]), 21)
    meta = load("step3b_activations_meta.json")
    c.add("probe: distinct hypotheses", meta["hypotheses"], 601)
    c.add("probe: median Spearman vs Cinà's OPT scores", meta["score_check_median_spearman"], 0.9998, 4)
    c.add("probe: minimum Spearman", meta["score_check_min_spearman"], 0.994, 3)
    c.add("T15 sentences in the training text", m["mapping"]["in_decoder_training_text"], 128)
    no = load("step1_sessions_nooverlap.json")
    c.add("T15 late sentences in the training text",
          m["temporal"]["late_sentences"] - no["temporal"]["late_sentences"], 117)
    c.add("  share of the 651 late sentences, %",
          100 * (m["temporal"]["late_sentences"] - no["temporal"]["late_sentences"]) / 651, 18, 0)
    c.add("T12 sentences in the training text", t["mapping"]["in_decoder_training_text"], 25)
    t_no = load("step1_sessions_t12_nooverlap.json")
    c.add("T12 late sentences in the training text",
          t["temporal"]["late_sentences"] - t_no["temporal"]["late_sentences"], 1)


def check_step1(c):
    from scipy.stats import spearmanr

    c.head("§3.1 Calibrated decoder: sessions (T15)")
    m = load("step1_sessions.json")
    p = m["P1_P3"]
    c.add("P1 hard/easy quartile ratio", p["P1_hard_over_easy_quartile_wer"], 11.4, 1)
    c.add("P1 hardest quartile WER", p["P1_hard_quartile_mean_wer"], 17.6, 1)
    c.add("P1 easiest quartile WER", p["P1_easy_quartile_mean_wer"], 1.5, 1)
    for arm, rho in (("opt", 0.39), ("qwen", 0.41), ("jev", 0.56)):
        c.add(f"P2 rho first pass vs {arm} gain", p[f"P2_spearman_firstpass_vs_{arm}_gain"]["rho"], rho, 2)
    sessions = [s for s in m["per_session"] if s["sentences"] >= 10 and s["first_pass_wer"] > 0]
    c.add("sessions with >= 10 sentences and any first-pass error", len(sessions), 37)
    for arm, rho in (("opt", 0.00), ("qwen", 0.20), ("jev", -0.12)):
        r = spearmanr([s["first_pass_wer"] for s in sessions],
                      [s[f"{arm}_gain"] / s["first_pass_wer"] for s in sessions])
        c.add(f"rho first pass vs {arm} share removed", r.correlation, rho, 2)
        c.add(f"  p > 0.2", bool(r.pvalue > 0.2), True)
    c.add("P3 rho date vs first pass", p["P3_spearman_date_vs_firstpass"]["rho"], 0.11, 2)
    c.add("P3 p", p["P3_spearman_date_vs_firstpass"]["p"], 0.52, 2)
    c.add("P4 largest |difference|", max(abs(v["difference"]) for v in m["P4"].values()), 0.06, 2)
    signs = {arm: v["temporal_split"]["eval_ece"] > v["random_split"]["eval_ece"] for arm, v in m["P5"].items()}
    c.add("P5 mixed in sign", len(set(signs.values())) == 2, True)

    c.head("§3.1 Calibrated decoder: churn on the random test set (T15)")
    h = load("step1_headroom.json")
    c.add("test words", h["test_words"], 6449)
    c.add("first pass WER", h["first_pass_wer"], 8.00, 2)
    for arm, fixed, broken, wer, split, oracle, headroom in (
            ("opt", 48, 29, 7.71, (41, 26, 22), 7.26, 0.45),
            ("qwen", 85, 63, 7.66, (70, 55, 43), 6.68, 0.98),
            ("jev", 121, 81, 7.38, (100, 73, 49), 6.12, 1.26)):
        a = h[arm]
        c.add(f"{arm} words fixed", a["words_fixed"], fixed)
        c.add(f"{arm} words broken", a["words_broken"], broken)
        c.add(f"{arm} WER", a["wer"], wer, 2)
        c.add(f"{arm} helped/harmed/neutral",
              (a["changes_helped"], a["changes_harmed"], a["changes_neutral"]), split)
        c.add(f"{arm} oracle-gate WER", a["oracle_gate_wer"], oracle, 2)
        c.add(f"{arm} headroom", a["wer"] - a["oracle_gate_wer"], headroom, 2)

    c.head("§3.1 Calibrated decoder: T12, held out by session")
    g = load("step3a_gate_t12.json")["temporal"]
    opt, jev = g["arms"]["opt"], g["arms"]["jev"]
    c.add("first pass WER", opt["wer"]["none"], 18.40, 2)
    c.add("OPT always-apply WER", opt["wer"]["always"], 18.40, 2)
    c.add("OPT helped/harmed", (opt["applied"]["always"]["helped"], opt["applied"]["always"]["harmed"]), (33, 30))
    c.add("Jev always-apply WER", jev["wer"]["always"], 18.61, 2)
    c.add("Jev helped/harmed", (jev["applied"]["always"]["helped"], jev["applied"]["always"]["harmed"]), (34, 36))
    c.add("OPT oracle WER", opt["wer"]["oracle"], 16.59, 2)
    c.add("Jev oracle WER", jev["wer"]["oracle"], 16.63, 2)
    c.add("OPT headroom (H2)", opt["wer"]["always"] - opt["wer"]["oracle"], 1.8, 1)
    c.add("Jev headroom (H2)", jev["wer"]["always"] - jev["wer"]["oracle"], 2.0, 1)
    c.add("Qwen alpha tuned on early sessions", g["alpha"]["qwen"], 0.0, 2)
    t = load("step1_sessions_t12.json")
    c.add("P1 ratio", t["P1_P3"]["P1_hard_over_easy_quartile_wer"], 2.6, 1)
    c.add("P2 rho, OPT", t["P1_P3"]["P2_spearman_firstpass_vs_opt_gain"]["rho"], 0.13, 2)
    c.add("P3 rho", t["P1_P3"]["P3_spearman_date_vs_firstpass"]["rho"], -0.37, 2)
    c.add("P3 p", t["P1_P3"]["P3_spearman_date_vs_firstpass"]["p"], 0.07, 2)
    c.add("P4 OPT |difference|", abs(t["P4"]["opt"]["difference"]), 0.41, 2)
    c.add("P4 Qwen |difference|", abs(t["P4"]["qwen"]["difference"]), 0.41, 2)
    c.add("P5 higher ECE late, all three arms",
          all(v["temporal_split"]["eval_ece"] > v["random_split"]["eval_ece"] for v in t["P5"].values()), True)


def check_step3a(c):
    c.head("§3.2 Table 1: T15, temporal split, 20 late sessions")
    g = load("step3a_gate.json")
    rows = {
        "opt": (7.75, 7.08, 7.15, 7.08, 6.63, (-0.23, 0.22), (75, 43, 18), (51, 33, 8)),
        "qwen": (7.75, 7.08, 7.08, 6.90, 6.21, (-0.42, 0.07), (119, 57, 34), (84, 49, 17)),
        "jev": (7.75, 6.90, 6.90, 6.90, 6.63, (0.00, 0.00), (63, 43, 10), (63, 43, 10)),
    }
    for arm, (none, always, soft, gate, oracle, ci, applied_always, applied_gate) in rows.items():
        a = g["temporal"]["arms"][arm]
        for policy, value in (("none", none), ("always", always), ("softmax_threshold", soft),
                              ("logistic_gate", gate), ("oracle", oracle)):
            c.add(f"{arm} {policy}", a["wer"][policy], value, 2)
        c.add(f"{arm} gate - always CI low", a["vs_always_ci"]["logistic_gate"][0], ci[0], 2)
        c.add(f"{arm} gate - always CI high", a["vs_always_ci"]["logistic_gate"][1], ci[1], 2)
        for policy, expected in (("always", applied_always), ("logistic_gate", applied_gate)):
            x = a["applied"][policy]
            c.add(f"{arm} {policy} applied (changes, helped, harmed)",
                  (x["changes"], x["helped"], x["harmed"]), expected)

    c.head("§3.2 Predictions Q1-Q4 and coefficients (T15, temporal)")
    arms = g["temporal"]["arms"]
    for arm, gain, share in (("opt", 0.00, 0), ("qwen", 0.18, 21), ("jev", 0.00, 0)):
        c.add(f"{arm} gate gain over always", arms[arm]["wer"]["always"] - arms[arm]["wer"]["logistic_gate"], gain, 2)
        c.add(f"{arm} share of headroom recovered, %", 100 * arms[arm]["share_of_headroom_recovered"], share, 0)
    c.add("Jev tuned threshold", arms["jev"]["threshold_logistic"], 0.0, 2)
    c.add("Q4 softmax within 0.1 of always, all arms",
          all(abs(a["wer"]["softmax_threshold"] - a["wer"]["always"]) <= 0.1 for a in arms.values()), True)
    for arm, feature, value in (("opt", "rescorer_agreement", 0.75), ("qwen", "length_words", 1.05),
                                ("jev", "edit_distance", 0.50)):
        coef = arms[arm]["standardized_coefficients"]
        c.add(f"{arm} top coefficient is {feature}", max(coef, key=lambda k: abs(coef[k])), feature)
        c.add(f"{arm} {feature} coefficient", coef[feature], value, 2)

    c.head("§3.2 Without training-text overlap (T15, temporal)")
    n = load("step3a_gate_nooverlap.json")["temporal"]["arms"]
    for arm, harm_gate, harm_always, wer_gate, wer_always in (("opt", 8, 16, 7.63, 7.58),
                                                              ("qwen", 13, 30, 7.39, 7.55)):
        c.add(f"{arm} harmful, gate", n[arm]["applied"]["logistic_gate"]["harmed"], harm_gate)
        c.add(f"{arm} harmful, always", n[arm]["applied"]["always"]["harmed"], harm_always)
        c.add(f"{arm} WER, gate", n[arm]["wer"]["logistic_gate"], wer_gate, 2)
        c.add(f"{arm} WER, always", n[arm]["wer"]["always"], wer_always, 2)

    c.head("§3.2 Random split (T15)")
    r = g["random"]["arms"]
    for arm, diff in (("opt", -0.73), ("qwen", -0.29), ("jev", -0.36)):
        c.add(f"{arm} gate - always", r[arm]["wer"]["logistic_gate"] - r[arm]["wer"]["always"], diff, 2)
    c.add("opt CI low", r["opt"]["vs_always_ci"]["logistic_gate"][0], -1.20, 2)
    c.add("opt CI high", r["opt"]["vs_always_ci"]["logistic_gate"][1], -0.33, 2)
    c.add("opt alpha, Cinà's dev split", load("step1_headroom.json")["opt"]["alpha"], 0.3, 2)
    c.add("opt alpha, dev split without ambiguous", g["random"]["alpha"]["opt"], 0.5, 2)
    c.add("opt test sentences changed", r["opt"]["test_changed"], 202)
    c.add("opt net gain", r["opt"]["wer"]["none"] - r["opt"]["wer"]["always"], 0.11, 2)

    c.head("§3.2 T12 replication (H1)")
    t = load("step3a_gate_t12.json")
    opt, jev = t["temporal"]["arms"]["opt"], t["temporal"]["arms"]["jev"]
    c.add("OPT harmful, gate vs always",
          (opt["applied"]["logistic_gate"]["harmed"], opt["applied"]["always"]["harmed"]), (11, 30))
    c.add("OPT helpful, gate vs always",
          (opt["applied"]["logistic_gate"]["helped"], opt["applied"]["always"]["helped"]), (16, 33))
    c.add("OPT gate WER", opt["wer"]["logistic_gate"], 18.19, 2)
    c.add("OPT gate - always CI low", opt["vs_always_ci"]["logistic_gate"][0], -0.90, 2)
    c.add("OPT gate - always CI high", opt["vs_always_ci"]["logistic_gate"][1], 0.47, 2)
    c.add("Jev threshold", jev["threshold_logistic"], 0.06, 2)
    c.add("Jev gate applies (changes, harmed)",
          (jev["applied"]["logistic_gate"]["changes"], jev["applied"]["logistic_gate"]["harmed"]), (94, 36))
    c.add("Qwen gate", t["temporal"]["arms"]["qwen"].get("gate"), "not fittable")
    c.add("OPT agreement coefficient", opt["standardized_coefficients"]["rescorer_agreement"], 0.99, 2)
    c.add("Jev agreement coefficient", jev["standardized_coefficients"]["rescorer_agreement"], 0.55, 2)
    for arm in ("opt", "jev"):
        coef = t["temporal"]["arms"][arm]["standardized_coefficients"]
        c.add(f"{arm} top coefficient", max(coef, key=lambda k: abs(coef[k])), "rescorer_agreement")
    tn = load("step3a_gate_t12_nooverlap.json")["temporal"]["arms"]
    c.add("no overlap: OPT harmful, gate vs always",
          (tn["opt"]["applied"]["logistic_gate"]["harmed"], tn["opt"]["applied"]["always"]["harmed"]), (12, 30))
    c.add("no overlap: Jev harmful, gate vs always",
          (tn["jev"]["applied"]["logistic_gate"]["harmed"], tn["jev"]["applied"]["always"]["harmed"]), (27, 33))
    rnd = t["random"]["arms"]
    c.add("random split: gate beats always, all three",
          all(a["wer"]["logistic_gate"] < a["wer"]["always"] for a in rnd.values()), True)
    q = rnd["qwen"]
    c.add("random split: Qwen gate - always", q["wer"]["logistic_gate"] - q["wer"]["always"], -0.65, 2)
    c.add("random split: Qwen CI low", q["vs_always_ci"]["logistic_gate"][0], -1.11, 2)
    c.add("random split: Qwen CI high", q["vs_always_ci"]["logistic_gate"][1], -0.19, 2)


def check_step3b(c):
    c.head("§3.3 Activation probe (OPT)")
    p = load("step3b_probe.json")
    t, r = p["temporal"], p["random"]
    c.add("probe layer", t["probe_last"]["layer"], 20)
    c.add("probe components", t["probe_last"]["components"], 8)
    c.add("probe WER", t["probe_last"]["wer"], 7.19, 2)
    c.add("gate WER", t["gate_3a"]["wer"], 7.08, 2)
    c.add("probe - gate", t["probe_last"]["minus_gate_3a"], 0.11, 2)
    c.add("probe - gate CI low", t["probe_last"]["minus_gate_3a_ci"][0], 0.04, 2)
    c.add("probe - gate CI high", t["probe_last"]["minus_gate_3a_ci"][1], 0.20, 2)
    c.add("AUC gate", t["gate_3a"]["test_auc_helped"], 0.63, 2)
    c.add("AUC probe", t["probe_last"]["test_auc_helped"], 0.64, 2)
    c.add("random: probe - gate", r["probe_last"]["minus_gate_3a"], 0.02, 2)
    c.add("random: CI low", r["probe_last"]["minus_gate_3a_ci"][0], -0.08, 2)
    c.add("random: CI high", r["probe_last"]["minus_gate_3a_ci"][1], 0.13, 2)
    c.add("random: AUC gate", r["gate_3a"]["test_auc_helped"], 0.79, 2)
    c.add("random: AUC probe", r["probe_last"]["test_auc_helped"], 0.81, 2)
    x = t["probe_mean_exploratory"]
    c.add("mean pooling applied (changes, helped, harmed)",
          (x["applied"]["changes"], x["applied"]["helped"], x["applied"]["harmed"]), (30, 22, 1))
    c.add("mean pooling WER", x["wer"], 7.21, 2)
    c.add("fit examples (OPT changes, fit set)", load("step3a_gate.json")["temporal"]["arms"]["opt"]["train_changed"], 90)


def per(rows):
    return 100 * sum(r["edits"] for r in rows) / sum(r["phonemes"] for r in rows)


def check_step2(c):
    from scipy.stats import spearmanr

    c.head("§3.4 Drift (phoneme error rate, %)")
    early = load("step2_per_early.json")["sessions"]
    released = {r["session"]: r for r in load("step2_per_released.json")["sessions"]}
    fit = [r for r in early if r["trained_on"]]
    late = [r for r in early if not r["trained_on"]]
    c.add("late sessions", len(late), 20)
    e_fit, r_fit = per(fit), per([released[r["session"]] for r in fit])
    c.add("D3 early model, fit sessions", e_fit, 10.05, 2)
    c.add("D3 released, fit sessions", r_fit, 9.22, 2)
    c.add("D3 relative difference, %", 100 * (e_fit / r_fit - 1), 9.1, 1)
    rho = spearmanr([r["days_after_training"] for r in late], [r["per"] for r in late])
    c.add("D1 rho", rho.correlation, 0.93, 2)
    c.add("D1 p, x 1e-9", rho.pvalue * 1e9, 4, 0)
    e_late, r_late = per(late), per([released[r["session"]] for r in late])
    c.add("D2 early model, late sessions", e_late, 56.3, 1)
    c.add("D2 released, late sessions", r_late, 11.2, 1)
    c.add("D2 ratio", e_late / r_late, 5.0, 1)
    by_day = {r["days_after_training"]: r for r in late}
    for day, value, ref in ((7, 14.8, 4.8), (12, 14.6, None), (14, 13.0, None), (21, 36.4, None), (103, 63.9, None)):
        c.add(f"day {day}, early model", 100 * by_day[day]["per"], value, 1)
        if ref is not None:
            c.add(f"day {day}, released", 100 * released[by_day[day]["session"]]["per"], ref, 1)
    year = [r for r in late if 411 <= r["days_after_training"] <= 504]
    c.add("day 411-504, early model min", 100 * min(r["per"] for r in year), 73.8, 1)
    c.add("day 411-504, early model max", 100 * max(r["per"] for r in year), 81.1, 1)
    c.add("day 411-504, released min", 100 * min(released[r["session"]]["per"] for r in year), 9.2, 1)
    c.add("day 411-504, released max", 100 * max(released[r["session"]]["per"] for r in year), 30.8, 1)
    near = [r for r in late if r["days_after_training"] <= 33]
    far = [r for r in late if r["days_after_training"] >= 91]
    c.add("near sessions", len(near), 5)
    c.add("near PER min", 100 * min(r["per"] for r in near), 13, 0)
    c.add("near PER max", 100 * max(r["per"] for r in near), 36, 0)
    c.add("far sessions", len(far), 15)
    c.add("far PER min", 100 * min(r["per"] for r in far), 42, 0)
    c.add("far PER max", 100 * max(r["per"] for r in far), 81, 0)


def check_step4(c):
    s = load("step4_rescoring.json")
    n = load("step4_rescoring_nooverlap.json")
    c.head("§3.5 Under drift: regimes and E0")
    c.add("near sentences", s["sentences"]["near"], 189)
    c.add("far sentences", s["sentences"]["far"], 462)
    c.add("E0 ours", s["E0"]["ours"], 7.82, 2)
    c.add("E0 Cinà", s["E0"]["cina"], 7.75, 2)
    c.add("E0 sentences", s["E0"]["sentences"], 651)
    c.add("E0 no overlap, ours", n["E0"]["ours"], 8.31, 2)
    c.add("E0 no overlap, Cinà", n["E0"]["cina"], 8.28, 2)
    c.add("E0 no overlap, sentences", n["E0"]["sentences"], 534)

    c.head("§3.5 Near regime: R1, R2")
    r = s["R1_R2"]
    c.add("first pass, drifted", r["opt"]["early_model"]["first_pass_wer"], 23.8, 1)
    c.add("first pass, released", r["opt"]["released_model"]["first_pass_wer"], 3.3, 1)
    c.add("OPT churn, drifted, %", 100 * r["opt"]["early_model"]["churn"], 34, 0)
    c.add("OPT churn, released, %", 100 * r["opt"]["released_model"]["churn"], 7, 0)
    c.add("OPT churn ratio", r["opt"]["churn_ratio"], 5.0, 1)
    c.add("Qwen churn, drifted, %", 100 * r["qwen"]["early_model"]["churn"], 37, 0)
    c.add("Qwen churn, released, %", 100 * r["qwen"]["released_model"]["churn"], 11, 0)
    for arm, gain, wer, gain_rel, share, share_rel in (("opt", 1.64, 22.2, 0.08, 6.9, 2.4),
                                                      ("qwen", 1.88, 22.0, -0.16, 7.9, -4.8)):
        c.add(f"{arm} gain, drifted", r[arm]["early_model"]["gain"], gain, 2)
        c.add(f"{arm} always WER, drifted", r[arm]["early_model"]["always_wer"], wer, 1)
        c.add(f"{arm} gain, released", r[arm]["released_model"]["gain"], gain_rel, 2)
        c.add(f"{arm} share, drifted, %", 100 * r[arm]["early_model"]["relative_gain"], share, 1)
        c.add(f"{arm} share, released, %", 100 * r[arm]["released_model"]["relative_gain"], share_rel, 1)
    c.add("R1 holds", s["R1_holds"], True)
    c.add("R2 holds", s["R2_holds"], False)

    c.head("§3.5 Far regime: R3, R4")
    c.add("far oracle", s["R3"]["far_oracle_wer"], 76.8, 1)
    c.add("near oracle", s["R3"]["near_oracle_wer"], 11.3, 1)
    c.add("fit-session oracle", s["R3"]["train_oracle_wer"], 2.5, 1)
    far_opt = s["R5"]["far"]["arms"]["opt"]["wer"]
    for policy, value in (("none", 89.7), ("always", 89.4), ("logistic_gate", 89.3), ("oracle", 88.5)):
        c.add(f"far OPT {policy}", far_opt[policy], value, 1)
    for src, reg, ratio, fixed, broken in ((s, "near", 1.88, 45, 24), (s, "far", 1.30, 39, 30),
                                           (n, "far", 2.00, 40, 20), (n, "near", 1.89, 51, 27)):
        tag = "" if src is s else "no overlap, "
        q = src["R4"][reg]
        c.add(f"{tag}{reg} fixed/broken", (q["words_fixed"], q["words_broken"]), (fixed, broken))
        c.add(f"{tag}{reg} ratio", q["fixed_to_broken"], ratio, 2)
    c.add("R3 holds", s["R3_holds"], True)
    c.add("R4 holds", s["R4_holds"], False)
    c.add("R4 holds without overlap", n["R4_holds"], True)
    c.add("far sentences in the training text", s["sentences"]["far"] - n["sentences"]["far"], 117)
    c.add("far sessions without overlap", len(n["sessions"]["far"]), 13)

    c.head("§3.5 Gate under drift: R5 (near regime)")
    a = s["R5"]["near"]["arms"]
    o, q = a["opt"], a["qwen"]
    c.add("OPT harmful, gate vs always",
          (o["applied"]["logistic_gate"]["harmed"], o["applied"]["always"]["harmed"]), (14, 15))
    c.add("OPT WER gate", o["wer"]["logistic_gate"], 22.36, 2)
    c.add("OPT WER always", o["wer"]["always"], 22.20, 2)
    c.add("OPT CI low", o["vs_always_ci"]["logistic_gate"][0], -0.20, 2)
    c.add("OPT CI high", o["vs_always_ci"]["logistic_gate"][1], 0.43, 2)
    c.add("OPT changes, gate vs always",
          (o["applied"]["logistic_gate"]["changes"], o["applied"]["always"]["changes"]), (59, 65))
    c.add("Qwen harmful, gate vs always",
          (q["applied"]["logistic_gate"]["harmed"], q["applied"]["always"]["harmed"]), (7, 16))
    c.add("Qwen helpful dropped / always helpful",
          (q["applied"]["always"]["helped"] - q["applied"]["logistic_gate"]["helped"], q["applied"]["always"]["helped"]),
          (12, 31))
    c.add("Qwen gate cost", q["wer"]["logistic_gate"] - q["wer"]["always"], 0.63, 2)
    c.add("Qwen WER gate", q["wer"]["logistic_gate"], 22.60, 2)
    c.add("Qwen WER always", q["wer"]["always"], 21.97, 2)
    c.add("Qwen CI low", q["vs_always_ci"]["logistic_gate"][0], 0.09, 2)
    c.add("Qwen CI high", q["vs_always_ci"]["logistic_gate"][1], 1.00, 2)
    c.add("R5 holds", s["R5_holds"], False)
    c.add("fit sentences in the training text", s["sentences"]["train"] - n["sentences"]["train"], 11)
    c.add("near test set unchanged without overlap", n["sentences"]["near"], s["sentences"]["near"])
    c.add("no overlap: OPT alpha", n["R5"]["near"]["alpha"]["opt"], 0.50, 2)
    c.add("full data: OPT alpha", s["R5"]["near"]["alpha"]["opt"], 0.45, 2)
    nq = n["R5"]["near"]["arms"]["qwen"]
    c.add("no overlap: Qwen gate cost", nq["wer"]["logistic_gate"] - nq["wer"]["always"], 0.39, 2)
    c.add("no overlap: Qwen WER gate", nq["wer"]["logistic_gate"], 22.36, 2)
    c.add("no overlap: Qwen harmful, gate vs always",
          (nq["applied"]["logistic_gate"]["harmed"], nq["applied"]["always"]["harmed"]), (4, 16))
    c.add("no overlap: Qwen CI low", nq["vs_always_ci"]["logistic_gate"][0], -0.20, 2)
    c.add("no overlap: Qwen CI high", nq["vs_always_ci"]["logistic_gate"][1], 0.80, 2)
    c.add("R5 holds without overlap", n["R5_holds"], True)
    c.add("verdict shift between the runs",
          (q["wer"]["logistic_gate"] - q["wer"]["always"]) - (nq["wer"]["logistic_gate"] - nq["wer"]["always"]), 0.23, 2)
    for arm, margin, agreement in (("opt", 0.55, -0.06), ("qwen", 0.78, 0.04)):
        coef = a[arm]["standardized_coefficients"]
        c.add(f"{arm} top coefficient", max(coef, key=lambda k: abs(coef[k])), "rescorer_margin")
        c.add(f"{arm} rescorer margin coefficient", coef["rescorer_margin"], margin, 2)
        c.add(f"{arm} agreement coefficient", coef["rescorer_agreement"], agreement, 2)
    c.add("Qwen zero-score wins, far", s["zero_score_wins"]["qwen"]["far"], 4, 0)
    c.add("zero-score wins elsewhere",
          sum(v for arm in ("opt", "qwen") for reg, v in s["zero_score_wins"][arm].items()
              if not (arm == "qwen" and reg == "far")), 0, 0)
    c.add("mean list length, drifted", s["lists"]["early"]["mean_list_len"], 60, 0)
    c.add("mean list length, released", s["lists"]["released_late"]["mean_list_len"], 40, 0)


def vetoes(arm):
    always, gate = arm["applied"]["always"], arm["applied"]["logistic_gate"]
    return (always["changes"] - gate["changes"], always["harmed"] - gate["harmed"], always["helped"] - gate["helped"])


class BuiltinsOnly(__import__("pickle").Unpickler):
    def find_class(self, module, name):
        raise __import__("pickle").UnpicklingError(f"refusing {module}.{name}")


def check_truth_in_list(c):
    """Share of sentences whose reference is among the drifted decoder's 100 candidates, per regime.
    Reads the committed pod output `derived/step4/lists_early.pkl` through a builtins-only unpickler."""
    from collections import defaultdict
    sys.path.insert(0, str(HERE / "src"))
    from sessions import normalize
    with open(HERE / "derived" / "step4" / "lists_early.pkl", "rb") as f:
        cache, _ = BuiltinsOnly(f).load()
    sessions = defaultdict(set)
    for row in load("val_sessions.json"):
        sessions[row["key"]].add(row["session"])
    regimes = load("step4_rescoring.json")["sessions"]
    c.head("§4 Truth in the drifted decoder's 100-best list")
    for regime, expected in (("near", (115, 189)), ("far", (6, 462))):
        members = set(regimes[regime])
        keys = [k for k in cache if len(sessions.get(k, ())) == 1 and next(iter(sessions[k])) in members]
        hits = sum(any(normalize(text) == k for text, *_ in cache[k]) for k in keys)
        c.add(f"{regime}: reference in list / sentences", (hits, len(keys)), expected)


def check_exploratory(c):
    c.head("§3.5 Exploratory: what the gate vetoes")
    t15 = load("step3a_gate.json")["temporal"]["arms"]
    t12 = load("step3a_gate_t12.json")["temporal"]["arms"]
    calibrated = [vetoes(t15["opt"]), vetoes(t15["qwen"]), vetoes(t12["opt"])]
    c.add("calibrated: vetoed, harmful, helpful", tuple(map(sum, zip(*calibrated))), (107, 46, 35))
    near = load("step4_rescoring.json")["R5"]["near"]["arms"]
    drifted = [vetoes(near["opt"]), vetoes(near["qwen"])]
    c.add("drifted near: vetoed, harmful, helpful", tuple(map(sum, zip(*drifted))), (39, 10, 15))


def check_discussion(c):
    c.head("§4 Discussion and limitations")
    t15 = load("step3a_gate.json")["temporal"]["arms"]
    t12 = load("step3a_gate_t12.json")["temporal"]["arms"]
    nets = [a["wer"]["none"] - a["wer"]["always"] for a in t15.values()]
    nets += [a["wer"]["none"] - a["wer"]["always"] for k, a in t12.items() if "wer" in a]
    c.add("net gain of rescoring, smallest", min(nets), -0.21, 2)
    c.add("net gain of rescoring, largest", max(nets), 0.85, 2)
    heads = [a["wer"]["always"] - a["wer"]["oracle"] for a in t15.values()]
    heads += [a["wer"]["always"] - a["wer"]["oracle"] for a in t12.values() if "wer" in a]
    c.add("oracle headroom, smallest", min(heads), 0.27, 2)
    c.add("oracle headroom, largest", max(heads), 2.0, 1)
    c.add("changed sentences per arm, fewest", min(a["test_changed"] for a in t15.values()), 63)
    c.add("changed sentences per arm, most", max(a["test_changed"] for a in t15.values()), 119)
    t12s = load("step3a_gate_t12.json")
    c.add("T12 Qwen alpha, random split", t12s["random"]["alpha"]["qwen"], 0.4, 1)


def check_review(c):
    c.head("Review checks (src/review_checks.py)")
    r = load("review_checks.json")
    w = r["P1"]["t15"]["word_weighted"]
    c.add("P1 word-weighted hardest quartile WER", w["hard"], 17.7, 1)
    c.add("P1 word-weighted easiest quartile WER", w["easy"], 1.8, 1)
    c.add("P1 word-weighted ratio", w["ratio"], 10.0, 1)
    o = r["early_model_overlap"]
    c.add("fit-set sentences in the released model's training text", o["fit_in_released_training_text"], 11)
    c.add("  of which in the early model's training sessions", o["fit_also_in_early_training"], 1)
    c.add("late sentences in the released model's training text", o["late_in_released_training_text"], 117)
    c.add("  of which in the early model's training sessions", o["late_also_in_early_training"], 0)
    q = r["qwen_zero_score"]
    c.add("far-regime lists whose top Qwen score is 0.0", q["far_lists_with_zero_top_score"], 4)
    c.add("  selected at the tuned alpha", q["selected_at_tuned_alpha"], 1)
    a = r["session_ambiguous_in_random_split"]
    c.add("session-ambiguous sentences", a["ambiguous"], 17)
    c.add("  in the 419-sentence development split", a["in_dev"], 5)
    c.add("  in the test split", a["in_test"], 12)
    c.add("development split size", a["dev"], 419)


def check_step5(c):
    c.head("§3.6 Controls asked for in review (S1, S3, S4)")
    r = load("step5_review.json")
    s1 = r["S1_near_stability"]
    c.add("S1 OPT gate minus always, near", s1["opt"]["full_gate_minus_always"], 0.16, 2)
    c.add("S1 OPT leave-one-out, lowest", min(s1["opt"]["leave_one_out"].values()), 0.00, 2)
    c.add("S1 OPT leave-one-out, highest", max(s1["opt"]["leave_one_out"].values()), 0.28, 2)
    c.add("S1 OPT per-session, lowest", min(v["gate_minus_always_points"] for v in s1["opt"]["per_session"].values()), -0.47, 2)
    c.add("S1 OPT per-session, highest", max(v["gate_minus_always_points"] for v in s1["opt"]["per_session"].values()), 0.60, 2)
    c.add("S1 Qwen gate minus always, near", s1["qwen"]["full_gate_minus_always"], 0.63, 2)
    c.add("S1 Qwen leave-one-out, lowest", min(s1["qwen"]["leave_one_out"].values()), 0.42, 2)
    c.add("S1 Qwen leave-one-out, highest", max(s1["qwen"]["leave_one_out"].values()), 0.76, 2)
    c.add("S1 Qwen per-session, highest", max(v["gate_minus_always_points"] for v in s1["qwen"]["per_session"].values()), 1.20, 2)
    s3 = r["S3_random_features"]
    c.add("S3 seeds", s3["seeds"], 20)
    c.add("S3 random minus gate, median", s3["minus_gate_3a"]["median"], 0.09, 2)
    c.add("S3 random minus gate, 5th percentile", s3["minus_gate_3a"]["p5"], 0.00, 2)
    c.add("S3 random minus gate, 95th percentile", s3["minus_gate_3a"]["p95"], 0.25, 2)
    c.add("S3 seeds worse than the gate", round(s3["minus_gate_3a"]["share_worse_than_gate"] * s3["seeds"]), 17)
    c.add("S3 seeds worse than the real probe", round(s3["share_of_seeds_worse_than_real"] * s3["seeds"]), 7)
    c.add("S3 real probe minus gate", s3["real_probe_minus_gate_3a"], 0.11, 2)
    s4 = r["S4_feature_distributions"]
    for arm in ("opt", "qwen"):
        c.add(f"S4 agreement helpful minus harmful, calibrated {arm}", s4["t15_calibrated"][arm]["features"]["rescorer_agreement"]["helpful_minus_harmful"], 0.76, 2)
    c.add("S4 agreement helpful minus harmful, near OPT", s4["t15_drift_near"]["opt"]["features"]["rescorer_agreement"]["helpful_minus_harmful"], 0.49, 2)
    c.add("S4 agreement helpful minus harmful, near Qwen", s4["t15_drift_near"]["qwen"]["features"]["rescorer_agreement"]["helpful_minus_harmful"], 0.56, 2)
    c.add("S4 near OPT, edit distance helpful minus harmful", s4["t15_drift_near"]["opt"]["features"]["edit_distance"]["helpful_minus_harmful"], -0.84, 2)
    c.add("S4 near OPT, accepted changes", s4["t15_drift_near"]["opt"]["accepted"], 59)
    c.add("S4 near OPT, changes", s4["t15_drift_near"]["opt"]["changes"], 65)
    ll = s4["list_length_matched_late_sessions"]
    c.add("S4 early model list length, late sessions", ll["early_model"]["mean"], 88.5, 1)
    c.add("S4 released model list length, late sessions", ll["released_model"]["mean"], 39.2, 1)
    c.add("S4 early model list length, all sessions", ll["early_model_all_sessions_mean"], 59.6, 1)

    c.head("§3.6 Controls asked for in review (S2, input layer)")
    il = load("step5_input_layer.json")
    m = il["late_mean_per"]
    c.add("S2 released, own input layer, late PER", 100 * m["released_own_layer"], 11.2, 1)
    c.add("S2 released, frozen 2023.11.26 layer, late PER", 100 * m["released_frozen_layer"], 45.6, 1)
    c.add("S2 early model, late PER", 100 * m["early_model"], 56.3, 1)
    c.add("S2 share of the gap carried by the input layer", (m["released_frozen_layer"] - m["released_own_layer"]) / (m["early_model"] - m["released_own_layer"]), 0.76, 2)
    by_day = {r["days_after_training"]: r for r in il["sessions"]}
    for day, frozen, own, early in ((7, 8.8, 4.8, 14.8), (12, 9.1, 3.5, 14.6), (14, 7.8, 2.9, 13.0), (21, 22.9, 8.6, 36.4)):
        c.add(f"S2 day {day}, frozen layer", 100 * by_day[day]["per_frozen_layer"], frozen, 1)
        c.add(f"S2 day {day}, own layer", 100 * by_day[day]["per_own_layer"], own, 1)
        c.add(f"S2 day {day}, early model", 100 * by_day[day]["per_early_model"], early, 1)


def rerun():
    from paths import CINA, DATA_ROOT
    if not (CINA / "data" / "lists_published.pkl").exists():
        sys.exit(f"--rerun needs Cinà's repository at {CINA} (set RESCORING_DRIFT_DATA; see README)")
    before = {p.name: p.read_bytes() for p in RESULTS.glob("*.json")}
    py = [sys.executable, "-I"]
    runs = []
    for participant in ("t15", "t12"):
        for overlap in ([], ["--exclude-train-overlap"]):
            args = ["--participant", participant, *overlap]
            runs += [["src/step1_sessions.py", *args], ["src/step3a_gate.py", *args]]
    runs += [["src/step1_headroom.py"], ["src/step4_rescoring.py"], ["src/step4_rescoring.py", "--exclude-train-overlap"]]
    env = dict(os.environ, RESCORING_DRIFT_DATA=str(DATA_ROOT))
    for cmd in runs:
        t0 = time.time()
        out = subprocess.run(py + cmd, cwd=HERE, env=env, capture_output=True, text=True)
        print(f"  {' '.join(cmd):60s} {time.time() - t0:6.1f}s  exit {out.returncode}")
        if out.returncode:
            print(out.stderr[-2000:])
            sys.exit(1)
    changed = [n for n, b in before.items() if (RESULTS / n).read_bytes() != b]
    print(f"\nresult files identical byte for byte: {len(before) - len(changed)}/{len(before)}")
    for n in changed:
        print(f"  changed: results/{n} (see `git diff results/{n}`)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true", help="re-run steps 1, 3a and 4 before checking")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "src"))
    if a.rerun:
        print("re-running steps 1, 3a and 4 from cached inputs")
        rerun()
    c = Checks()
    for check in (check_data_and_splits, check_step1, check_step3a, check_step3b, check_step2, check_step4,
                  check_exploratory, check_discussion, check_truth_in_list, check_review, check_step5):
        check(c)
    ok = c.report()
    print("\nNot checked here (GPU, neural data or model weights): step 2 training (RTX 4090, 190.8 min),\n"
          "step 2 PER evaluation (neural data), step 3b activations (OPT-6.7b), the step 4 decode and scoring\n"
          "(RTX 3090, 3-gram LM). The README gives each command.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
