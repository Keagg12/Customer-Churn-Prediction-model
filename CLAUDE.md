# Permanent instructions: Churn Prediction Project

This is the only set of active instructions. A new Claude receives: this file, the latest `HANDOFF.md`, and the current project code files, attached by the user. It has no access to earlier chats, the user's computer or VS Code. Rely only on what is attached.

## Project
Explainable AI-Based Customer Churn Prediction and Personalized Marketing Recommendation System (Predict → Explain → Recommend). This is a **standalone software product by an independent developer**, not a college or academic project. **The sole goal is the working, well-tested application.**

Scope rule: the earlier PPT, synopsis, technical overview and project description are reference material for understanding intended functionality only. Do not edit, generate or maintain PPTs, synopses, reports or other documents unless the user explicitly asks. Any such files in `deliverables/`, `docs/` or `originals/` are old reference and need not be carried over.

## Branding rules
No college, department, mentor, team, roll-number or academic branding anywhere. Do not ask the user about institutional details. Use neutral, professional product branding. Do not invent a company, developer identity, customers, deployments or business results. The files in `originals/` are old academic drafts; never use them as a branding source.

## Principles
- Build the actual project. Keep documentation minimal: `HANDOFF.md` is the only state record.
- Make routine technical decisions independently. Ask only when something important is missing or scope would change.
- Never publish, upload, push or spend anything without the user's approval. No secrets or personal data in files.

## Starting a session
1. Read this file and `HANDOFF.md`. Inspect every attached file.
2. Reconcile them: the attached code is the truth for what exists; the user's statements are the truth for intent. Report real discrepancies briefly.
3. Continue from the handoff's next action. Do not restart, redo finished work, or reopen settled decisions without a reason.

## Delivering code
- The user applies code in VS Code by hand. Give **complete** file contents, one block per file, each headed with its exact relative path and marked **NEW** or **REPLACEMENT** (a replacement always contains the whole updated file, never a patch or "rest unchanged").
- Give the user a short list of commands to run and what output to expect.
- Do not create ZIPs or ask for the whole project folder.
- Keep a file manifest in `HANDOFF.md` (see below), updated whenever files change.

## Status words (use exactly)
- **Generated:** written by Claude, not yet confirmed applied.
- **Applied:** the user says they put it in their project.
- **Tested:** the user pasted real output (or Claude ran it in its own sandbox and says so) showing it works.
Never say the application "works" without evidence. Say what was run, where, and the observed result. Claude's sandbox results are not the user's VS Code results.

## Dataset checks (dataset not yet received; expected 30 Oct 2026)
Never assume its format, columns, labels or results. Do no dataset-dependent work before inspecting the real files. Then answer, in plain language:
1. Can it be read and understood?
2. Is there a usable churn label, or enough to define one?
3. Are data quality and class balance reasonable?
4. Can we train and properly evaluate a churn model (no leakage, sensible split)?
5. Can we explain predictions (understandable features for SHAP)?
6. Can we make useful retention recommendations from the available features?
Verdict: **proceed** (go straight to implementation), **minor adjustments** (state the issue and the simplest fix), or **reconsider part of the project** (explain evidence and options; scope changes need user approval).
Report only results from real runs. The 30%/70% risk thresholds are provisional until checked against real score distributions. SHAP shows what the model relies on, not proven causes.

## Checkpoint (when asked, or before stopping)
Update `HANDOFF.md` fully (it must stand alone), then tell the user which files to carry over to the next account: this file, `HANDOFF.md`, and every code file listed in the manifest.
