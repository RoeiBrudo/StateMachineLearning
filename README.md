# StateMachineLearning

This repository is a collection of independent research/experiment subprojects around **learning state machines / automata** and related sequence models.

Each **top-level folder** is a separate subproject.

## Subprojects (ideas and goals)

## 1) Automata - Binary

**Folder:** `Automata - Binary/`

A minimal, educational implementation of **active automata learning** over a **binary alphabet**.

- The target “language” is implemented as a simple rule (`Language`), and the learner interacts with it through an **oracle** (`ORC`) that supports:
  - membership queries (“what is the label of this prefix/word?”)
  - equivalence queries that return a **counterexample prefix** when the hypothesis automaton disagrees
- The learner maintains a **classification tree** (`TreeMng.py`) and refines the hypothesis automaton until it matches the oracle on the tested space.

## 2) Automata - MultiClassifier

**Folder:** `Automata - MultiClassifier/`

A generalization of the binary experiment to:

- **non-binary alphabets** (arbitrary symbols in `ALF`)
- **multi-class outputs** (each learned state is mapped to a class/label)

Conceptually: it learns a finite-state model that, given a sequence of symbols, predicts a class—using the same active-learning pattern (oracle + counterexamples + refinement).

## 3) HMM&SEQ2AUTOMATA

**Folder:** `HMM&SEQ2AUTOMATA/`

A pipeline that bridges **probabilistic sequence models** and **finite automata**.

High-level idea:

- Start from real-world (or provided) tabular/time-series data.
- **Discretize** continuous features/outputs into clusters (to get a symbolic view).
- Use an HMM-based learner to model/predict.
- Convert the discretized behavior into an **automaton** (via an Angluin-style learner inside `AngluinClassifier/`).

This is useful when you want an interpretable, finite-state abstraction of a model’s behavior.

## 4) HMMR

**Folder:** `HMMR/`

An implementation of an **HMM with regression emissions** (“HMMR”):

- Hidden state evolves via a Markov chain.
- The observed output `Y` is produced by a **state-specific regression model**.

The code includes the usual building blocks:

- sampling/generating sequences from a model
- likelihood computation (forward/backward in log-space)
- Viterbi decoding
- parameter export/import (CSV files)

Overall: it’s a focused playground for learning and evaluating this hybrid “discrete state + continuous regression” model family.

## 5) Model2Automata

**Folder:** `Model2Automata/`

A more general “**extract an automaton from a model**” pipeline.

Core idea:

- Take a base predictive model (some included models are neural / TensorFlow-based).
- **Discretize inputs** (cluster feature vectors) and **discretize predictions/hidden states**.
- Treat discretized sequences as words over an alphabet.
- Learn an automaton whose states correspond to stable behavioral modes and whose transitions follow the discretized dynamics.

This subproject is geared towards producing a finite-state surrogate that approximates a complex model.

## 6) shuffle ideals learning

**Folder:** `shuffle ideals learning/`

A standalone implementation related to learning a target sequence/pattern in the context of **shuffle ideals** (see the included PDF).

It focuses on:

- generating samples
- using statistical queries to iteratively recover the target pattern symbol-by-symbol

This project is more theory/algorithm oriented than the others.

## 7) transfer framework

**Folder:** `transfer framework/`

Experiments around **transfer / multi-task learning** and sample complexity.

Key comparison:

- A shared-parameter **multi-learner** (shared representation + task-specific bias)
- Versus **classic** baselines that learn separate models per task

The goal is to study how performance changes as:

- the amount of data grows
- the number of tasks/classifiers grows

## Notes

- Not all subprojects share a single unified dependency setup.
- This README intentionally focuses on *what each subproject is about*, not how to run it.
