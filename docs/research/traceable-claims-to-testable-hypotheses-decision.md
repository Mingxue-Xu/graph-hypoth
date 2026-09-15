# From Traceable Claims to Testable Hypotheses

**Thesis.** GraphHypothesize separates model-generated assessments from the decisions that enter the research record, and makes that passage inspectable. This is process integrity. It does not establish that an extracted claim is faithful, that an evidence assessment is calibrated, that a hypothesis is novel, or that an experiment plan is informative. Those are empirical questions that the record makes easier to ask and does not answer.

**Scope and conventions.** The findings come from the public implementation, representative exported artifacts, and primary literature. Implementation statements describe the public workflow and its shipped defaults; some concern configuration-dependent paths. External studies motivate evaluations and supply designs; none of their results measures GraphHypothesize. Each finding is stated as observed behaviour, its scientific implication, and the evaluation it calls for. The evaluations are developed in the [Research Agenda](research-agenda.md); the design rationale is in [Claims Before Causes](claims-before-causes.md).

## 1. The objects and what a check can establish

Concepts are nodes and stated relationships are directed edges. The graph is structured at the software level, but relationship labels are free-form. An edge therefore records an assertion or a proposal, not an identified causal effect. In the claim-seeded workflow, an agent extracts concepts and relationships; retrieved literature is scored and selected for the edges scheduled for review; an agent produces evidential sub-signals that deterministic rules map to a verdict and a confidence; further checks decide whether the verdict can be recorded. Hypothesis development uses graph structure and literature to propose candidates, which pass model critique and deterministic filtering before a confirmation gate. That gate is human oversight only when a person makes the selection. Committed hypotheses can enter experiment planning; plans are checked for required content and resolvable citations and may record that no relevant methods support was found.

Two facts about the flow matter for everything below. The graph view supplied to the hypothesis generator omits the recorded edge verdicts and confidence values, so the design should not be described as a loop in which verified relationships guide generation. And every stage's checks are structural.

**Table 1. Distinct meanings of validation in the pipeline.**

| Object | What an implemented check can establish | What requires additional evidence |
|---|---|---|
| Graph change | The proposal satisfies schema, reference, state-transition and duplicate-application rules | The represented claim preserves the intended scientific meaning |
| Evidence assessment | Preserved sub-signals are mapped to a verdict by specified rules | The verdict is accurate and its confidence is calibrated |
| Hypothesis candidate | The candidate meets configured criteria and was selected for commitment | It is novel, plausible and useful to investigate |
| Experiment plan | Required content is present and declared grounding IDs resolve within the available evidence scope | The cited passages support the plan, and the design is feasible and informative |
| Research record | Recorded sources and decisions can be inspected to the extent preserved | The complete run can be reconstructed and unauthorised changes detected |

## 2. Findings by subsystem

### 2.1 Preserving the meaning of a claim

*Observed.* Content-derived identities and transaction records give a stable basis for distinguishing objects and inspecting accepted and rejected changes, and authored priority is separated from verification state. But labels are free-form, so paraphrases such as "reduces" and "decreases" can create distinct edges while a broad label can hide differences in direction or scope. The schema holds textual conditions, confounders and scope, but no dedicated fields for effect sign and magnitude or a consistently structured population, intervention, comparator and outcome. Initial extraction uses a single model response with structural parsing; its auxiliary quality measures are not a general semantic admission check, and the concept canonicalization available during enrichment is not applied at extraction. Node merging can leave evidence or plan references attached to retired identities.

*Implication.* A structurally accepted graph can omit a qualification or split one concept across records, after which every downstream stage appraises an assertion that differs from the one the researcher intended. Opposing edges may be a genuine disagreement or different experimental conditions.

*Evaluation.* Whether researchers recover the same qualified proposition from the input and the extracted graph, examining omitted conditions, altered modality or negation, relation direction and duplicate concepts separately. Claimify treats ambiguity, coverage and decontextualisation as explicit extraction concerns, and EDC separates extraction from canonicalization; they motivate the evaluation without establishing that either method transfers <sup>[5](#reference-5), [6](#reference-6)</sup>.

### 2.2 Retrieving evidence that can change the assessment

*Observed.* Cross-source deduplication, recorded retrieval inputs and deterministic association scoring make the reviewed material inspectable. Relevance signals over a fixed SPECTER2 representation <sup>[8](#reference-8)</sup> are combined with uncalibrated weights, and every evidence role draws on one claim-level pool obtained by topical search, with no coverage estimate. The default scheduling policy requests support-role evidence only. The verdict architecture also represents contradiction, qualification and a non-causal reading, but recording those verdicts needs evidence carrying the matching roles, so with support-only links they cannot pass the role requirement. This limits the default workflow, not the verdict vocabulary.

*Implication.* Topical similarity retrieves papers about the same relationship without necessarily finding the strongest challenge to it, and deduplicating records does not identify several papers that report one study. The workflow constrains which conclusions can be recorded without showing how often it would accept a claim wrongly.

*Evaluation.* Relevant-evidence recall and stance coverage measured separately on a bounded corpus with reference evidence, then a test of whether supplying omitted counterevidence changes the verdict appropriately. SciFact's separation of evidence retrieval from support and refutation judgments is the methodological reference <sup>[7](#reference-7)</sup>. Retrieval failure must stay distinguishable from backend failure and from a genuine absence of informative studies.

### 2.3 Mapping evidence to conclusions and revising them

*Observed.* Deterministic fusion makes the decision rule explicit, and abstention lets the system withhold a conclusion, so a disagreement can be located in the evidence, the sub-signals or the rule. Nothing establishes calibration: weights and thresholds are configured, with no demonstrated relation between reported confidence and observed correctness. The state-transition rules make supported, contradicted, qualified and non-causal verdicts terminal through the verification interface; an insufficient verdict can move to a substantive one, but a settled verdict cannot be reopened by the same mechanism.

*Implication.* Confidence values are outputs of the appraisal rule, not probabilities that a claim is true. POPPER's sequential falsification with Type-I error control addresses a different verification problem from deterministic literature appraisal, and the two should not be equated <sup>[3](#reference-3)</sup>. A null finding needs its effect estimate, uncertainty and test sensitivity represented at the study level before it can contradict, qualify or leave a claim unresolved. Terminal verdicts restrict the response to new evidence or a corrected interpretation; temporal knowledge graphs such as Zep show invalidation with retained history as one alternative <sup>[9](#reference-9)</sup>.

*Evaluation.* Calibration on independently labelled claim–evidence sets, with acceptance policies compared at matched coverage as in selective evaluation <sup>[22](#reference-22)</sup>, and revision under controlled additions or corrections to evidence. Preserving earlier decisions and revising current assessments are compatible requirements.

### 2.4 Evaluating hypotheses beyond fluent novelty judgments

*Observed.* Generation, critique and commitment are separate stages. The critic panel receives candidate structure and grounding material and excludes the generator's private rationale and self-grades; its judgments are aggregated by deterministic rules. Candidate eligibility combines model judgments with embedding-based comparisons, most components of the composite ranking are generator-provided, and the panel adds a field-novelty judgment and a literature-saturation assessment. Model diversity across judges is not enforced, a third judge is consulted only on selected disagreements, candidate order is shared across judges, and numerical grades lack range checks. The median of two grades equals their mean.

*Implication.* Distance from existing concept descriptions is not the same construct as novelty of a relationship, and novelty is not plausibility, testability or research value; a saturation penalty can demote a worthwhile replication. Agreement among judges cannot reveal a mistake they share, and shared ordering confounds candidate identity with position.

*Evaluation.* Expert judgments on the full candidate pool with origin and scores hidden, novelty separated into concepts, relationships, mechanisms and applications, replication value included, and presentation counterbalanced. RINoBench and RQ-Bench report substantial divergence between model and expert novelty judgments in their settings <sup>[10](#reference-10), [11](#reference-11)</sup>; PoLL supports diverse panels in the tasks it studies <sup>[12](#reference-12)</sup>, while nine judges have been found to supply about two independent votes' worth of information <sup>[13](#reference-13)</sup>. Tournament ranking is a candidate intervention, not an established remedy.

### 2.5 Turning a hypothesis into an informative test

*Observed.* Plans make operationalisation, comparisons, measurements, expected outcomes and falsification criteria explicit; declared grounding IDs must resolve within the retrieved material; and a plan can record the absence of relevant methods evidence. Admission checks do not establish that a supplied quotation occurs verbatim in its source or that it justifies the method. Model-based review imposes no numerical acceptance threshold, a final refinement is not necessarily reviewed again, and the procedure field is not required. A representative run holds five plans, two marked as having no relevant methods, committed beside ten unverified edges: proposals were produced and recorded, and nothing validated them.

*Implication.* Structural acceptance and model review are not scientific validation. A detailed plan can be feasible yet weakly informative when its predicted observations are also expected under plausible alternatives.

*Evaluation.* Independent researchers judging executability, source support and which alternatives a plan's results could distinguish, followed by execution of selected plans. BED-LLM motivates an explicit information-gain objective, but its demonstrated tasks are question selection and preference elicitation, so transfer is a proposal <sup>[14](#reference-14)</sup>. A hypothesis need not be confirmed for its experiment to be useful if the system can incorporate an informative negative result.

### 2.6 Maintaining an inspectable and revisable research record

*Observed.* The transaction core assigns content-based identities, checks changes before acceptance and records accepted and rejected proposals. The examined graph-state path records events without linking them through predecessor hashes, and ordinary exported receipts do not contain the transaction payloads the replay routine needs. The node-merge operation recomputes edge identities without remapping associated evidence links and plans. Graph-pipeline prompts lack the explicit instruction–data framing present in the web-retrieval prompts.

*Implication.* Traceable references help inspect a result but do not by themselves show complete reconstruction of graph history or detection of altered records, and merges can break the continuity a later audit depends on. Retrieved text reaches model-based appraisal, so indirect prompt injection is a live concern: it has been demonstrated against model-based scientific reviewers <sup>[16](#reference-16)</sup>, and a prompt warning is not an enforced boundary, as CaMeL's architectural separation of control flow from untrusted data makes clear <sup>[15](#reference-15)</sup>.

*Evaluation.* Whether an independent investigator can reconstruct which claim was assessed, which passage was available, how the assessment was derived and what later evidence changed it, using only an export, including runs with merges and rejected changes; and resistance to injected instructions tested architecturally rather than by prompt wording.

## 3. Position among scientific-discovery systems

The comparison is organised by documented activity and form of evaluation and assigns no ranking, because literature appraisal, idea assessment, data analysis and laboratory testing share no outcome measure in the evidence considered. A capability omitted from a paper is not assumed absent from its implementation.

**Table 2. Documented focus and the evidential distinction that matters here.**

| System | Documented focus | Distinction |
|---|---|---|
| GraphHypothesize | Controlled changes to a claim graph; evidence appraisal; critique; experiment proposals | Process behaviour is established; scientific accuracy is unmeasured |
| ResearchAgent <sup>[1](#reference-1)</sup> | Literature-informed idea generation with iterative reviewing agents | Idea-quality evaluations differ from executing the research |
| SciAgents <sup>[2](#reference-2)</sup> | Ontological graph reasoning and multi-agent hypothesis development | Case studies illustrate ideation; experimental support needs separate evidence |
| The AI Scientist <sup>[17](#reference-17)</sup> | Idea generation, code, computational experiments, manuscripts, automated review | Successful execution does not establish correctness of every reported interpretation |
| Kosmos <sup>[18](#reference-18)</sup> | Iterative data analysis and literature research through a world model | Its expert statement audit measures output accuracy, a form of evidence distinct from process checks |
| Co-Scientist <sup>[19](#reference-19)</sup> | Generation, critique and tournament evolution of hypotheses | Biomedical validations support selected applications, not every ranking |
| POPPER <sup>[3](#reference-3)</sup> | Executed falsification tests with sequential statistical control | Type-I guarantees concern the testing framework, not literature synthesis |
| Robin <sup>[20](#reference-20)</sup> | Hypothesis generation and analysis connected to laboratory feedback | Semi-autonomous; scientists perform the experiments |
| Coscientist <sup>[4](#reference-4)</sup> | Tool-assisted planning and execution of chemistry experiments | Physical execution, bounded by the reported chemistry tasks |

Four external results set expectations for the evaluations above. They measure other systems and tasks, not GraphHypothesize.

| Study | Result | What it motivates here |
|---|---|---|
| Kosmos expert audit <sup>[18](#reference-18)</sup> | 79.4% of 102 statements judged accurate; 57.9% for synthesis statements | A per-statement audit is the nearest precedent for measuring output accuracy |
| RINoBench <sup>[10](#reference-10)</sup> | Best zero-shot macro-F1 of 17.2; models avoided the "not novel" category | Validating the panel's novelty judgments against experts |
| Novelty-judge study <sup>[11](#reference-11)</sup> | Judge–judge agreement 52%→62% under prompt revision; expert–judge agreement 22%→30–40% | Model agreement is not expert agreement |
| Scientific-agent traces <sup>[21](#reference-21)</sup> | Across more than 25,000 runs, traces ignored available evidence 68% of the time and revised beliefs 26%; the base model explained 41.4% of outcome variance, the scaffold 1.5% | Measuring evidence use and revision directly rather than assuming scaffolding secures them |

A testable architectural direction is a persistent claim layer beneath an execution-capable system such as POPPER or Robin. That study would need an explicit mapping from proposed hypotheses to tested predictions, procedures, observations and revised assessments, and would have to preserve disagreements and negative findings. Compatibility has not been demonstrated.

## 4. What to evaluate

The findings call for seven studies: semantic fidelity of extraction, exposure to counterevidence, calibrated acceptance, warranted revision, hypothesis selection, experimental informativeness, and researcher utility against an equally complete evidence dossier. The [Research Agenda](research-agenda.md) develops four of them as a chain (counterevidence, calibration, revision with execution feedback, researcher benefit) and lists the other three as deferred, together with the foundations they depend on.

## 5. Limitations

This assessment produces no new performance estimates. The implementation inspection was focused, not exhaustive, and some properties vary with configuration or execution path. The representative run separates generated artifacts from scientific validation; it is not representative and cannot estimate failure prevalence. No model or scientific experiment was rerun. External results motivate concerns and interventions without determining how often the corresponding failures occur here, and source selection was question-driven rather than systematic.

## References
<a id="reference-1"></a>
1. Baek, J., Jauhar, S. K., Cucerzan, S., & Hwang, S. J. (2025). [ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models](https://arxiv.org/abs/2404.07738). NAACL.
<a id="reference-2"></a>
2. Ghafarollahi, A., & Buehler, M. J. (2024). [SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](https://arxiv.org/abs/2409.05556). arXiv:2409.05556.
<a id="reference-3"></a>
3. Huang, K., Jin, Y., Li, R., Li, M. Y., Candès, E., & Leskovec, J. (2025). [Automated Hypothesis Validation with Agentic Sequential Falsifications (POPPER)](https://arxiv.org/abs/2502.09858). ICML; arXiv:2502.09858.
<a id="reference-4"></a>
4. Boiko, D. A., MacKnight, R., Kline, B., & Gomes, G. (2023). [Autonomous chemical research with large language models](https://www.nature.com/articles/s41586-023-06792-0). Nature, 624, 570–578.
<a id="reference-5"></a>
5. Metropolitansky, D., & Larson, J. (2025). [Towards Effective Extraction and Evaluation of Factual Claims (Claimify)](https://arxiv.org/abs/2502.10855). ACL.
<a id="reference-6"></a>
6. Zhang, B., & Soh, H. (2024). [Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction (EDC)](https://aclanthology.org/2024.emnlp-main.548/). EMNLP, 9820–9836.
<a id="reference-7"></a>
7. Wadden, D., et al. (2020). [Fact or Fiction: Verifying Scientific Claims (SciFact)](https://aclanthology.org/2020.emnlp-main.609/). EMNLP, 7534–7550.
<a id="reference-8"></a>
8. Singh, A., et al. (2023). SciRepEval / SPECTER2: Scientific Document Representations.
<a id="reference-9"></a>
9. Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/html/2501.13956v1). arXiv:2501.13956.
<a id="reference-10"></a>
10. Schopf, T., & Färber, M. (2026). [Is This Idea Novel? An Automated Benchmark for Judgment of Research Ideas (RINoBench)](https://arxiv.org/abs/2603.10303). arXiv:2603.10303; accepted to LREC 2026.
<a id="reference-11"></a>
11. Sinhahajari, S., Majumder, N., & Poria, S. (2026). [On the Limits of LLM-as-Judge for Scientific Novelty Assessment (RQ-Bench)](https://arxiv.org/abs/2606.12071). arXiv:2606.12071.
<a id="reference-12"></a>
12. Verga, P., et al. (2024). [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL)](https://arxiv.org/abs/2404.18796). arXiv:2404.18796.
<a id="reference-13"></a>
13. [Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels](https://arxiv.org/abs/2605.29800). arXiv:2605.29800 (2026).
<a id="reference-14"></a>
14. Choudhury, D., et al. (2026). [BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design](https://arxiv.org/abs/2508.21184). ICLR.
<a id="reference-15"></a>
15. Debenedetti, E., et al. (2025). [Defeating Prompt Injections by Design (CaMeL)](https://arxiv.org/abs/2503.18813). arXiv:2503.18813.
<a id="reference-16"></a>
16. Sahoo, S., et al. (2025). [When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection](https://arxiv.org/abs/2512.10449). arXiv:2512.10449.
<a id="reference-17"></a>
17. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292). arXiv:2408.06292.
<a id="reference-18"></a>
18. Mitchener, L., et al. (2025). [Kosmos: An AI Scientist for Autonomous Discovery](https://arxiv.org/abs/2511.02824). arXiv:2511.02824.
<a id="reference-19"></a>
19. Gottweis, J., et al. (2026). [Accelerating scientific discovery with Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y). Nature, 655, 487–496.
<a id="reference-20"></a>
20. Ghareeb, A. E., et al. (2026). [A multi-agent system for automating scientific discovery (Robin)](https://www.nature.com/articles/s41586-026-10652-y). Nature, 655, 497–505.
<a id="reference-21"></a>
21. Ríos-García, M., et al. (2026). [AI Scientists Produce Results Without Reasoning Scientifically](https://arxiv.org/abs/2604.18805). arXiv:2604.18805.
<a id="reference-22"></a>
22. Jung, J., Brahman, F., & Choi, Y. (2025). [Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](https://arxiv.org/abs/2407.18370). ICLR 2025.
