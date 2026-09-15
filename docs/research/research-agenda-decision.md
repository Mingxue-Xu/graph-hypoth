# Research Agenda

**Goal, in fifteen words:** Help researchers generate, evaluate, test, and revise scientific hypotheses using traceable evidence and calibrated uncertainty. This is the intended goal, not a description of what the system achieves. The shipped workflow implements extraction, literature appraisal, hypothesis development, confirmation and experiment-plan recording; execution, calibrated judgment and demonstrated researcher benefit are still to be established.

**How to read this.** Four claims are developed below. Each states the current public implementation, the literature and its limits, an addition to evaluate, and the evidence of progress that would count. The additions are proposals inferred from literature and code; none has been measured in GraphHypothesize. The four form a chain: expose claims to counterevidence (A), calibrate what is accepted (B), revise conclusions when evidence changes, including evidence from executed experiments (C), and show that the resulting record helps a researcher (D). Other directions are listed at the end with the reason each is deferred. The ranking is editorial, not a result the literature supplies.

## A. Expose claims to counterevidence

**Current implementation.** The role vocabulary already covers support, contradiction, qualification, mechanism and confounder, the verification cycle can appraise mixed-role evidence together, and it requires a matching role before committing any substantive verdict ([verification cycle](../../src/cycles/verification.py)). But the shipped workflow schedules only the support role ([run_path.py](../../src/run_path.py)), and the live dependency builder supplies the same claim-level evidence pool to every target ([graph_state_runtime.py](../../src/graph_state_runtime.py)), so selecting another role performs no new targeted search. A computed contradict, qualify or not-causal verdict therefore lacks the role it needs and is recorded as an open risk rather than as a verdict. [Claims Before Causes](claims-before-causes.md) calls this the reachability gap. It is the cheapest fix in the agenda: the machinery exists and the schedule starves it.

**Literature and its limits.** SciFact separates evidence retrieval, stance and rationale selection, which makes retrieval failure and appraisal failure measurable apart <sup>[1](#reference-1)</sup>. OpenScholar is a scientific retrieval-and-synthesis comparator <sup>[2](#reference-2)</sup>. Yoon et al. show that generated query expansions can already contain information the reference evidence supports, a confound that any generated challenge query inherits <sup>[3](#reference-3)</sup>. None establishes complete coverage of the literature relevant to a live hypothesis.

**Addition to evaluate.** Connect role scheduling to a bounded retrieval policy that actively searches for contradiction, qualification, causal alternatives and methodological challenges. Preserve the original claim, the generated queries, the returned passages and the sources that could not be accessed as separate records, with unexpanded retrieval as a comparator. Combine the relevant roles before commitment. Record that a challenge search found nothing without reading that as evidence that no counterevidence exists, and keep "no meaningful effect", "explicit denial of causation" and "inconclusive estimate" distinguishable.

**Evidence of progress.** Compare the shared-pool workflow with targeted challenge retrieval under the same initial claim, total query budget and reading budget, on a curated collection annotated for support, contradiction, scope restriction, methodological criticism and irrelevance, with evidence grouped by underlying study. Measure role coverage and accepted-claim error. Add a supplied-evidence condition, in which the curated counterevidence is handed to the appraiser directly, to separate failure to find a challenge from failure to interpret one. More queries or more documents alone would not support the claim; if counterevidence is recovered but ignored, the problem is appraisal or commitment, not retrieval.

## B. Calibrate acceptance against an explicit target

**Current implementation.** Evidence appraisal records sub-signals, and deterministic formulas combine support, counterevidence, breadth, risks and qualifiers into a confidence that the selector can refuse as insufficient when scores are low or close ([verification cycle](../../src/cycles/verification.py)). These are real controls over consistency and abstention. The current tests pin the configured constants and behaviour; a test named "calibrated" does not estimate calibration against independent outcomes ([test_graph_config_defaults.py](../../tests/unit/test_graph_config_defaults.py)). No empirical calibration result was located. The same fact bounds the local-model question: a backend change shifts score distributions, and fixed thresholds then move verdicts, which [Which Agents Go Local?](which-agents-go-local.md) lists as score drift.

**Literature and its limits.** Trust or Escalate offers the selective-evaluation design: calibrate when to accept a model judgment and when to abstain or escalate, with a statistical target of agreement with human preferences under stated assumptions <sup>[4](#reference-4)</sup>. Its guarantees do not turn a scientific score into a probability of truth and do not hold under unrestricted domain shift.

**Addition to evaluate.** Define the target before fitting anything. A defensible target is agreement with independent expert adjudication of whether a claim is supported by the supplied evidence, which differs from the probability that the claim is true in the world. Evaluate the existing deterministic scores first as scores. If they are converted to probabilities, fit the mapping on development data, freeze the acceptance policy, and evaluate on held-out papers and domains, keeping paper families in one partition. Keep expert uncertainty and disagreement visible instead of recoding them as errors. Escalate difficult cases to an independent reviewer with the cost recorded.

**Evidence of progress.** Hold candidate judgments fixed and vary the acceptance policy: the existing threshold, calibrated selection, and escalation. Report error among accepted judgments with the fraction accepted, a risk–coverage curve, out-of-domain and later-evidence degradation, and the cost of escalation. A policy that accepts almost nothing is reliable and useless. Repeating a model judgment does not solve the problem, because replication of a shared error raises apparent certainty. A useful negative result is no improvement over a simple threshold at matched coverage and cost.

## C. Revise conclusions and their dependents, with execution feedback

**Current implementation.** An insufficient verdict can move to a substantive verdict; supported, contradicted, qualified and not-causal are otherwise terminal through the verification transition rules ([validator.py](../../src/validator.py)). Attaching an experiment plan does not change the claim's verdict. The graph-state experiment stage retrieves methods, designs and optionally refines plans, and commits them against hypothesis edges with distinct grounding states for cited methods, retrieved-but-irrelevant methods and none retrieved ([experiment cycle](../../src/cycles/experiment.py)). It does not execute those plans, store an outcome as a dedicated result record, or feed an outcome back through reassessment. The loop is open at both ends: nothing revises a settled verdict, and nothing produces the observation that would warrant revising it.

**Literature and its limits.** DeReLab supplies formally generated defeasible-reasoning updates, including exceptions and irrelevant information, and BayesBench distinguishes updating beliefs from using them in later predictions <sup>[5](#reference-5), [6](#reference-6)</sup>; their formal reference models do not decide how much weight a heterogeneous scientific study deserves. On execution, POPPER connects measurable implications to sequential falsification with Type-I control <sup>[7](#reference-7)</sup>, ReplicatorBench evaluates replication workflows including non-replicable cases <sup>[8](#reference-8)</sup>, and Robin and Co-Scientist report selected biomedical follow-up <sup>[9](#reference-9), [10](#reference-10)</sup>. All are bounded by domain, expert involvement and statistical assumptions.

**Addition to evaluate.** Introduce a governed reassessment transaction that retains the prior verdict, the changed evidence, the rationale and the new assessment, and distinguishes status revision, scope narrowing, source withdrawal and an unchanged conclusion. Record claim dependencies so that a correction flags the hypotheses and plans built on the corrected premise. Then add an execution interface and durable result records linking the tested claim, the final protocol, data and code versions, quality controls, observed estimates, uncertainty and interpretation, starting with approved, bounded computational experiments; laboratory work needs an execution partner and domain controls. Any sequential-testing layer must show that each generated implication is logically appropriate for the hypothesis, handle adaptive data reuse and selection across many hypotheses, and state the scope of its error control.

**Evidence of progress.** Compare incremental revision with preserved history against full recomputation from the same accumulated evidence, on sequences whose desired update is independently specified: genuine contradiction, a narrower population, duplicated evidence, withdrawal of a source, and irrelevant new information, with confirming and disconfirming updates matched for strength and with order varied while the final evidence set is held constant. Measure whether required revisions occur, whether unaffected claims stay stable, and whether dependent hypotheses and plans are reconsidered. For execution, prespecify separate denominators for proposals generated, plans selected, executions attempted, results interpretable and claims updated; a quality-controlled null with adequate sensitivity is informative, and a failed assay or script is not falsification. Check that the resulting updates agree with independent analysis before describing the system as a feedback loop.

## D. Demonstrate researcher benefit against an equally complete dossier

**Current implementation.** The workflow exports graph state, evidence assessments, surfaced hypotheses, experiment plans and audit material ([run_path.py](../../src/run_path.py)), and unit and fixture tests check that these mechanisms behave. Nothing measures whether a researcher reaches a better scientific decision with them. No controlled researcher-utility result was located.

**Literature and its limits.** DiscoveryBench assesses data-driven hypotheses, ScienceAgentBench scientific analysis tasks, TruthInsightBench evidence-linked artifacts, and K-Bench delivered work on real scientific requests <sup>[11](#reference-11), [12](#reference-12), [13](#reference-13), [14](#reference-14)</sup>. Their units and reference judgments differ, and a model-graded benchmark does not supply an independent validity label merely because its aggregation is deterministic.

**Addition to evaluate.** Build a domain-stratified evaluation set with qualified source claims, relevant challenges, candidate pools and selected executable tasks. Then run a researcher study with three conditions: a final report alone, an equally complete flat evidence dossier, and the dossier plus the claim graph and its revision history. The dossier–graph contrast isolates the organisation of evidence from the benefit of receiving more of it. Match model access, reading time and tool budgets; counterbalance task order; never give an investigator the same underlying problem twice when carryover would reveal the answer.

**Evidence of progress.** Tasks should require locating support for a claim, identifying a scope mismatch, distinguishing independent studies from repeated reports, explaining a revision, and determining which downstream proposals a correction affects. Primary outcomes are independently adjudicated task answers, correction of overbroad claims, error localisation and investigator time; perceived clarity is secondary. Report disagreement, abstention and failed runs with uncertainty at the research-problem level. If the graph condition does not beat the dossier, traceability may still be useful, but the claim that graph organisation improves research reasoning stays unproven.

## Foundations, deferred claims, and work order

**Foundations.** Three lower-ranked claims are prerequisites for measuring A–D credibly: the graph must preserve a claim's meaning through extraction and merging, a cited passage must warrant the specific use made of it, and the exported record must let another investigator reconstruct decisions through merges and corrections. [From Traceable Claims to Testable Hypotheses](traceable-claims-to-testable-hypotheses.md) lists the specific gaps: single-call extraction without semantic admission, quotations not checked against sources at plan admission, exports without transaction payloads, events without predecessor hashes, and merges that do not remap evidence. These are engineering work with a clear specification rather than open research questions, and they come first.

**Deferred.** Each row below was one of the twelve claims or one of the later directions. They are deferred because they depend on A–D, because their literature concerns judge validity rather than this design, or because they broaden scope beyond the system.

| Claim | One-line statement | Feeds | Anchor literature |
|---|---|---|---|
| Evidence-aware generation | Give the Synthesist premise assessments and let it mark which premises it accepts, doubts or deliberately challenges | A, C | SciMON; Si et al. <sup>[19](#reference-19), [20](#reference-20)</sup> |
| Discriminating tests | Require explicit alternatives and predicted observations under each before a plan is execution-ready | C | BioPlanner; BED-LLM <sup>[21](#reference-21), [22](#reference-22)</sup> |
| Valuable selection | Evaluate novelty, plausibility, testability and replication value separately, blind to origin, and audit judge agreements | D | RINoBench; RQ-Bench; PoLL; Geometry of LLM-as-Judge; HindSight <sup>[23](#reference-23), [24](#reference-24), [25](#reference-25), [26](#reference-26), [27](#reference-27)</sup> |
| Meaning preservation | Source-aligned claim records with polarity, modality, direction and scope, compared with the source before commitment | foundation | Claimify; EDC; Hagen et al.; Lu et al. <sup>[28](#reference-28), [29](#reference-29), [30](#reference-30), [31](#reference-31)</sup> |
| Citation warrant | One citation audit at extraction, commitment and final plan review that keeps quotation, paraphrase and extrapolation distinct | foundation | CLAIM-BENCH; SourceCheckup <sup>[32](#reference-32), [33](#reference-33)</sup> |
| Reconstructable provenance | Full transaction exports, stable identities across merges, and a tested integrity mechanism if tamper detection is a goal | foundation | PROV-AGENT <sup>[34](#reference-34)</sup> |
| Independent corroboration | Distinguish document, study and observation so repeated reports do not count as replications | B | CONFACT <sup>[35](#reference-35)</sup> |
| Verified-outcome routing | Allocate model computation by cost per verified completion rather than per call | B | RouteLLM <sup>[36](#reference-36)</sup> |
| Joint retrieval and reader evaluation | Evaluate retriever, reader and delivery interface together, since rankings reverse across readers | A | Sen et al. <sup>[37](#reference-37)</sup> |
| Communication fidelity | Test that reader-facing rewrites preserve polarity, uncertainty and numbers, not only fluency | D | Hayakawa et al.; xCOMET <sup>[38](#reference-38), [39](#reference-39)</sup> |

**The same problems elsewhere.** The four claims are not specific to science. A debugging assistant, a policy-document analyst and a decision-support tool all derive decisions from evidence and face the same failures. The methods worth borrowing are truth maintenance and incremental view maintenance for revising derived conclusions when inputs change (C) <sup>[15](#reference-15)</sup>, selective prediction and conformal risk control for acceptance policies tied to a stated loss (B) <sup>[16](#reference-16)</sup>, diversified retrieval for surfacing evidence that could overturn a decision (A) <sup>[17](#reference-17)</sup>, and controlled human–AI decision studies that separate speed, appropriate reliance and decision quality (D) <sup>[18](#reference-18)</sup>. Transfer in either direction is a hypothesis, and the scientific workflow is not already a general decision-support platform.

**Work order.** Do the foundations, then A, B, C, D. A comes first because it is cheapest and because its evaluation produces the adjudicated cases B needs; C needs B's acceptance policy to decide when a revision is warranted; D is last because it tests the record the others produce. The smallest coherent extension connects source-aligned claims, deliberate challenge retrieval, inspectable reassessment and complete provenance, then tests that chain on a bounded set of independently adjudicated cases. Preregister each study's primary outcome and acceptable trade-offs before evaluating held-out tasks, and report failures, uncompleted cases and human effort. The goal does not require maximising graph size, novelty scores, the number of agents or the number of committed plans.

## References
<a id="reference-1"></a>
1. Wadden, D., et al. (2020). [Fact or Fiction: Verifying Scientific Claims (SciFact)](https://aclanthology.org/2020.emnlp-main.609/). EMNLP, 7534–7550.
<a id="reference-2"></a>
2. Asai, A., et al. (2026). [Synthesizing scientific literature with retrieval-augmented language models (OpenScholar)](https://www.nature.com/articles/s41586-025-10072-4). Nature.
<a id="reference-3"></a>
3. Yoon, Y., Jung, J., Yoon, S., & Park, K. (2025). [Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion](https://aclanthology.org/2025.findings-acl.980/). Findings of ACL 2025.
<a id="reference-4"></a>
4. Jung, J., Brahman, F., & Choi, Y. (2025). [Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](https://arxiv.org/abs/2407.18370). ICLR 2025.
<a id="reference-5"></a>
5. Sadhu, J., Shahad, S., & Marino, K. (2026). [DeReLab: Probing Defeasible Reasoning and Confirmation Bias in LLMs with a Generative Benchmark](https://arxiv.org/abs/2608.30413). arXiv:2608.30413; authors report acceptance to EMNLP 2026.
<a id="reference-6"></a>
6. Samanta, A., et al. (2026). [BayesBench: Evaluating LLM Belief Trajectories Under Multi-Turn Evidence Accumulation](https://arxiv.org/abs/2606.30850). arXiv:2606.30850.
<a id="reference-7"></a>
7. Huang, K., Jin, Y., Li, R., Li, M. Y., Candès, E., & Leskovec, J. (2025). [Automated Hypothesis Validation with Agentic Sequential Falsifications (POPPER)](https://arxiv.org/abs/2502.09858). ICML; arXiv:2502.09858.
<a id="reference-8"></a>
8. Nguyen, B., et al. (2026). [ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences](https://arxiv.org/abs/2602.11354). arXiv:2602.11354; KDD 2026 AI4Sciences acceptance reported by the authors.
<a id="reference-9"></a>
9. Ghareeb, A. E., et al. (2026). [A multi-agent system for automating scientific discovery (Robin)](https://www.nature.com/articles/s41586-026-10652-y). Nature, 655, 497–505.
<a id="reference-10"></a>
10. Gottweis, J., et al. (2026). [Accelerating scientific discovery with Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y). Nature, 655, 487–496.
<a id="reference-11"></a>
11. Majumder, B. P., et al. (2025). [DiscoveryBench: Towards Data-Driven Discovery with Large Language Models](https://arxiv.org/abs/2407.01725). ICLR 2025.
<a id="reference-12"></a>
12. Chen, Z., et al. (2025). [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://arxiv.org/abs/2410.05080). ICLR 2025.
<a id="reference-13"></a>
13. Yang, Z., Zhang, C., Zhang, Y., & Wang, H. (2026). [TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents](https://arxiv.org/abs/2609.05079). arXiv:2609.05079.
<a id="reference-14"></a>
14. Brueckner, A. M., Patel, D., He, Y., & Kassis, T. (2026). [K-Bench: measuring model performance on real scientific agent requests](https://arxiv.org/abs/2608.21601). arXiv:2608.21601.
<a id="reference-15"></a>
15. Doyle, J. (1979). [A Truth Maintenance System](https://www.sciencedirect.com/science/article/pii/0004370279900080). Artificial Intelligence.
<a id="reference-16"></a>
16. Angelopoulos, A. N., et al. (2024). [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html). ICLR.
<a id="reference-17"></a>
17. Goldstein, J., & Carbonell, J. (1998). [Summarization: (1) Using MMR for Diversity-Based Reranking and (2) Evaluating Summaries](https://aclanthology.org/X98-1025/). TIPSTER workshop.
<a id="reference-18"></a>
18. Kayser, M., et al. (2024). [Fool Me Once? Contrasting Textual and Visual Explanations in a Clinical Decision-Support Setting](https://aclanthology.org/2024.emnlp-main.1051/). EMNLP.
<a id="reference-19"></a>
19. Wang, Q., Downey, D., Ji, H., & Hope, T. (2024). [SciMON: Scientific Inspiration Machines Optimized for Novelty](https://arxiv.org/abs/2305.14259). ACL 2024.
<a id="reference-20"></a>
20. Si, C., Yang, D., & Hashimoto, T. (2025). [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](https://arxiv.org/abs/2409.04109). ICLR 2025.
<a id="reference-21"></a>
21. O'Donoghue, O., et al. (2023). [BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology](https://arxiv.org/abs/2310.10632). EMNLP 2023.
<a id="reference-22"></a>
22. Choudhury, D., et al. (2026). [BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design](https://arxiv.org/abs/2508.21184). ICLR.
<a id="reference-23"></a>
23. Schopf, T., & Färber, M. (2026). [Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas (RINoBench)](https://arxiv.org/abs/2603.10303). arXiv:2603.10303; accepted to LREC 2026.
<a id="reference-24"></a>
24. Sinhahajari, S., Majumder, N., & Poria, S. (2026). [On the Limits of LLM-as-Judge for Scientific Novelty Assessment (RQ-Bench)](https://arxiv.org/abs/2606.12071). arXiv:2606.12071.
<a id="reference-25"></a>
25. Verga, P., et al. (2024). [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL)](https://arxiv.org/abs/2404.18796). arXiv:2404.18796.
<a id="reference-26"></a>
26. Mukherjee, S., Hamna, H., Bali, K., & Sitaram, S. (2026). [The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment](https://arxiv.org/abs/2606.03043). arXiv:2606.03043; authors report acceptance to EMNLP 2026.
<a id="reference-27"></a>
27. Jiang, B. (2026). [HindSight: Evaluating LLM-Generated Research Ideas via Future Impact](https://arxiv.org/abs/2603.15164). arXiv:2603.15164.
<a id="reference-28"></a>
28. Metropolitansky, D., & Larson, J. (2025). [Towards Effective Extraction and Evaluation of Factual Claims (Claimify)](https://arxiv.org/abs/2502.10855). ACL.
<a id="reference-29"></a>
29. Zhang, B., & Soh, H. (2024). [Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction (EDC)](https://aclanthology.org/2024.emnlp-main.548/). EMNLP, 9820–9836.
<a id="reference-30"></a>
30. Hagen, T., et al. (2026). [Investigating Counterclaims in Causality Extraction from Text](https://arxiv.org/abs/2510.08224). arXiv:2510.08224.
<a id="reference-31"></a>
31. Lu, Y., Ziems, N., Dang, H., & Jiang, M. (2025). [Optimizing Decomposition for Optimal Claim Verification](https://aclanthology.org/2025.acl-long.254/). ACL 2025.
<a id="reference-32"></a>
32. Javaji, S. R., et al. (2025). [Can AI Validate Science? Benchmarking LLMs on Claim→Evidence Reasoning in AI Papers (CLAIM-BENCH)](https://aclanthology.org/2025.ijcnlp-long.127/). IJCNLP-AACL 2025, 2355–2379.
<a id="reference-33"></a>
33. Wu, K., et al. (2025). [An automated framework for assessing how well LLMs cite relevant medical references (SourceCheckup)](https://www.nature.com/articles/s41467-025-58551-6). Nature Communications.
<a id="reference-34"></a>
34. Souza, R., et al. (2025). [PROV-AGENT: Unified Provenance for Tracking AI Agent Interactions in Agentic Workflows](https://arxiv.org/abs/2508.02866). IEEE eScience 2025.
<a id="reference-35"></a>
35. Ge, Z., et al. (2025). [Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs (CONFACT)](https://www.ijcai.org/proceedings/2025/1073). IJCAI 2025.
<a id="reference-36"></a>
36. Ong, I., et al. (2025). [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665). arXiv:2406.18665.
<a id="reference-37"></a>
37. Sen, S., Kasturi, Lumer, Gulati, & Subbiah (2026). [Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](https://arxiv.org/abs/2605.15184). arXiv:2605.15184.
<a id="reference-38"></a>
38. Hayakawa, A., Bott, S., & Saggion, H. (2025). [Towards Trustworthy Lexical Simplification: Exploring Safety and Efficiency with Small LLMs](https://aclanthology.org/2025.inlg-main.15/). INLG.
<a id="reference-39"></a>
39. Guerreiro, N. M., et al. (2024). [xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection](https://aclanthology.org/2024.tacl-1.54/). TACL.
