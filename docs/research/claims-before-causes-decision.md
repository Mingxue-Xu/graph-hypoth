# Claims Before Causes

**Thesis.** GraphHypothesize stores what sources assert, not what is causally true. That is the right object for literature-grounded hypothesis work today, and the deterministic core that governs it earns its cost by controlling durable state, not by producing better hypotheses. This document explains the choice, appraises the implementation that makes it, and states the conditions under which a claim could be promoted to a cause.

**Scope and conventions.** The argument draws on the public source tree, the shipped configuration defaults, the pipeline and guide documentation, a representative exported run, and the causal-inference, causal-discovery, LLM-causality and evidence-synthesis literatures. Statements about the code describe the public checkout and its shipped defaults. External figures are reported as read and were not independently reproduced. The full-length version of this argument, with its 93-item bibliography, is [Claims Before Causes](claims-before-causes.md).

## 1. Why a claim graph and not a causal graph

A Pearl-style causal graph makes ontic claims: an edge asserts a structural dependency, and the graph supports intervention reasoning only as part of a specified causal model whose variables are well defined <sup>[1](#reference-1), [2](#reference-2)</sup>. Learning or checking such a graph invokes the Markov and faithfulness assumptions, and causal sufficiency wherever latent confounding is excluded <sup>[2](#reference-2)</sup>. A *Strict Causal Graph* (SCG) here means a context-specific DAG or SCM whose variables, arrows, assumptions and intervention semantics are specified well enough to answer stated identification queries.

The system stores something else: "source S, in context C, asserts X→Y, with this provenance, appraised this way." Generated hypotheses are recorded separately and may lack evidence links. The schema is a ledger of causal assertions. An SCM could be written over it, but the inputs neither justify its structure nor support identification or estimation. The deterministic core enforces structural integrity and traceability; it does not establish semantic fidelity or causal truth. The trade is preferable for auditability, and it has costs: "supported" never means "causal"; the shipped configuration can commit only two of the five verification verdicts (§3); and the ladder from claims to causes (§4) is designed but not climbed.

Five lines of evidence say the causal-graph preconditions fail, severally and jointly.

- **Variables are not well defined.** Nodes are content-addressed text labels with an open-string type. Causal-abstraction theory requires even curated aggregates to be proven to behave as macro-variables <sup>[3](#reference-3)</sup>; Hernán and Robins require an intervention version for every cause <sup>[4](#reference-4)</sup>, which literature sentences rarely specify and which a 2025 audit of applied epidemiology finds routinely unevaluated <sup>[5](#reference-5)</sup>. There is no data plane, so Markov, faithfulness and sufficiency cannot be checked, and faithfulness is only partly testable in principle <sup>[6](#reference-6)</sup>. The source tree contains no do-operator, structural equation, adjustment set, d-separation or acyclicity check: A→B and B→A can both commit.
- **Extractors are unreliable at exactly this task.** SemRep reaches 0.69 precision and 0.42 recall under relaxed evaluation, with entity normalisation about 27% of analysed errors <sup>[7](#reference-7)</sup>. CauseNet reports about 83% precision over eleven million relations and describes itself as a graph of "claimed causal relations" <sup>[8](#reference-8)</sup>. (The corrected reading of the PubMedCausal figures is in [Which Agents Go Local?](which-agents-go-local.md).)
- **Language models do not close the gap.** Kıcıman et al. report up to 97% pairwise direction accuracy and still position LLMs as assistants rather than inference engines <sup>[9](#reference-9)</sup>. Zečević et al.'s "causal parrots" diagnosis holds that models recite causal talk from training text <sup>[10](#reference-10)</sup>; in Corr2Cause none of seventeen models performed well and fine-tuned gains collapsed under variable renaming <sup>[11](#reference-11)</sup>; Feng et al. tie causal-discovery performance to pretraining frequency <sup>[12](#reference-12)</sup>. An LLM-authored "causal graph" would be a claim graph with its provenance stripped off.
- **Statistical discovery would not rescue it.** Benchmark success of continuous structure learners is largely a varsortability artefact of simulated DAGs <sup>[13](#reference-13)</sup>; across sixteen domain-shift tasks, predictors restricted to expert-selected causal features generalised no better than predictors using all features <sup>[14](#reference-14)</sup>; the field's own 2025 assessment concedes missing real-world ground truth <sup>[15](#reference-15)</sup>.
- **Aggregating claims is not causal inference.** Greenberg's dissection of 242 papers shows citation alone converting hypothesis into fact <sup>[16](#reference-16)</sup>; positive results are cited about 1.6× more often and hypothesis-confirming papers about 2.7× <sup>[17](#reference-17)</sup>. Significance-based vote counting is unreliable and direction-of-effect vote counting is a limited option <sup>[18](#reference-18)</sup>; GRADE rates certainty in a body of evidence for an outcome, so applying it to single graph assertions is a project-specific adaptation <sup>[19](#reference-19)</sup>. "N papers assert this edge" is Cartwright's "no causes in, no causes out" in operational form <sup>[20](#reference-20)</sup>, and health-science causation needs both difference-making and mechanistic evidence <sup>[21](#reference-21)</sup>.

The project's own design survey points the same way: every validated building block it cites is claim-native, from FActScore's atomic claims <sup>[22](#reference-22)</sup> and POPPER's falsification over natural-language hypotheses <sup>[23](#reference-23)</sup> to Kosmos's cited-statement world model, 79.4% of whose statements independent scientists found accurate <sup>[24](#reference-24)</sup>, and Heuer's Analysis of Competing Hypotheses <sup>[25](#reference-25)</sup>. Methods with formal model-discrimination guarantees, such as Hainy et al.'s, require specified candidate generative models, priors and simulators that the literature does not supply <sup>[26](#reference-26)</sup>.

**What the choice solves, leaves unsolved, and gained.**

| | |
|---|---|
| **Solved** | Honest semantics: an edge is an assertion with a settlement status, and the reader card is forced to say "This is a testable idea, not a proven conclusion." Provenance that resists spoofing: cache-key columns are authoritative over metadata, quote resolution fails closed, and linked evidence IDs must resolve. LLM containment: models propose, fixed arithmetic decides, parsers fail closed, and the overclaim guard refuses verdicts whose required evidence role was never retrieved. Human authority kept separate: priority is authored, never inferred, and hypotheses commit only on explicit confirmation. Plan objects whose expected and falsifying outcomes can seed intervention design. |
| **Not solved** | Identification, by design: "supported, confidence 0.8" summarises literature appraisal and estimates nothing about an intervention. The reachability gap, by configuration (§3). Causal content held as prose: polarity, effect size, dose, population and context are untyped, and contradictory literature is often context underspecification the graph cannot index <sup>[27](#reference-27)</sup>. Citation and publication bias enter uncorrected, with breadth capped at three distinct sources. An open loop: plans are prose deliverables that nothing executes or feeds back. |
| **Gained** | A durably exported, versioned graph state backed during execution by a replay-capable transaction log (exact replay needs the full transaction rows plus the external model and retrieval inputs). The state already carries fields a future causal layer needs: mechanism text, confounder and mediator nodes, scope qualifiers, plan objects. No integrated system in the surveyed literature takes a literature-derived claim graph through to a causal graph; a provenance-preserving claim graph is the state of the art (INDRA <sup>[28](#reference-28)</sup>, Kosmos <sup>[24](#reference-24)</sup>) and a defensible foundation for the extension. |

## 2. Why causal claims and deterministic measurement at all

Contemporary hypothesis generators carry neither commitment. MOOSE-Chem rediscovers published hypotheses that a model judge scores against a gold answer <sup>[29](#reference-29)</sup>; HypoBench scores hypotheses by held-out predictive accuracy <sup>[30](#reference-30)</sup>; TruthHypo scores them against relations that later entered a knowledge graph <sup>[31](#reference-31)</sup>; the co-scientist ranks free-text hypotheses by an Elo tournament <sup>[32](#reference-32)</sup>. The implemented design gives five reasons the pipeline is different. They concern state, measurement and cost, not hypothesis quality.

1. **The pipeline commits durable state, and a commit boundary must be deterministic.** The implemented architecture makes the durable product a causal claim graph plus transaction log, with graph validation rather than a model controller deciding what persists. Generate-then-verify is reliable exactly where the check is structural or executable, which is the LLM-Modulo position that models generate and external verifiers decide <sup>[33](#reference-33)</sup>. The five accept gates (schema, references, base hash, allowed transition, idempotency) and verbatim quote matching are checks of that class. The failure they guard against is common: in the MAST taxonomy, absent or incomplete verification (8.2%) and incorrect verification (9.1%) together account for 17.3% of annotated multi-agent failures <sup>[34](#reference-34)</sup>.
2. **Verification is target-scoped, and the targets are causal because the questions are.** Retrieval and appraisal attach to one node or edge and one evidence role at a time. The roles are support, contradiction, qualification, confounder and mechanism, and the verdict set includes `not_causal`, a label with no precedent in claim-verification datasets that the implementation keeps as a deliberate product choice: the system is meant to record that a relation is associational but not causal, which a co-occurrence graph cannot express. Typed, directional predications also let mechanistic knowledge compose and be checked for contradiction, as INDRA's executable models require <sup>[28](#reference-28)</sup>.
3. **The deliverable is an experiment, and an experiment is a causal object.** Woodward's definition of a direct cause, an intervention on X that changes Y with the other variables held fixed, is close to a template for the plan schema: intervention, comparator, controls, predicted and falsifying outcomes <sup>[35](#reference-35)</sup>. Rival explanations must be causal for discriminating experiments to be designed; controls are drawn from confounder- and mediator-typed neighbours; and a researcher's own results return as evidence through the same verification path.
4. **Committed decisions must not flip between runs, and model judgments do.** The runtime policy is committed-decision invariance, not bit-determinism. Every gate, score and verdict is a pure function of recorded, rubric-anchored ordinal sub-signals; the signals are recorded for replay; calibration work can estimate cross-run flip rates; near-ties abstain; human confirmation is the terminal abstention; and "numbers are calibration proposals, not truth." Evidence synthesis adopted the same posture against reviewer drift by fixing appraisal methods before results are seen <sup>[36](#reference-36)</sup>, and the evidence on model judgment supports keeping verdict authority outside the model: intrinsic self-correction does not improve reasoning <sup>[37](#reference-37)</sup>, and model evaluators recognise and favour their own generations <sup>[38](#reference-38)</sup>.
5. **The output moves resources, so precision and abstention are rational.** Under the field's cost asymmetry <sup>[39](#reference-39)</sup>, Chow's rule makes abstention rational whenever an error costs more than no answer <sup>[40](#reference-40)</sup>, while default model training rewards a confident guess over an abstention <sup>[41](#reference-41)</sup>. The confidence floor, the near-tie hold and the overclaim guard engineer the abstention that training does not. The accountability regime requires a human able to trace every AI-assisted claim to its basis, or the result is an illusion of understanding <sup>[42](#reference-42)</sup>. The counter-evidence bounds the scope rather than refuting it: Si et al. found that blinded experts rate LLM-generated ideas more novel than human ones <sup>[43](#reference-43)</sup>, so precision machinery belongs where a verdict or plan could commit resources, which is where the system places it, leaving the Synthesist free-form and gating verification and commit.

The benchmarks need none of this: they score resemblance to an existing reference once, commit no state and attach no cost to error. Where the discipline is needed it sits in the evaluation layer, as in POPPER's sequential test <sup>[23](#reference-23)</sup> and the project's frozen `fhq-v1` protocol, which compares GraphHypothesize with selected baselines on TOMATO-Chem by blinded expert preference.

**What the evidence does not show.** It does not show that causal typing raises the generation hit rate: Swanson's founding discoveries were made from co-occurrence, with the mechanism supplied by a human reader <sup>[44](#reference-44)</sup>. It does not show that determinism confers validity: composite scores over unreliable inputs can reverse a meta-analysis's conclusion <sup>[45](#reference-45)</sup>, and the calibration work needed to estimate flip rates has not yet been reported. Reliability is necessary for validity, not sufficient. These are questions for `fhq-v1`.

## 3. The implementation as built

**Strengths.** Verdict authority is deterministic, versioned and replayable. Confidence is the fixed formula Conf(v) = 0.55·Top3 + 0.25·Balance + 0.20·Breadth − 0.25·OpenRisk − 0.15·Limits over clamped model sub-signals, where Top3 is the mean of up to the three strongest supporting signals, Balance compares support with counterevidence strength, Breadth is the capped distinct-source count, and OpenRisk and Limits are capped counts of unresolved risks and qualifying limitations; the constants live in versioned defaults modules. Abstention is a first-class outcome (0.5 floor, 0.025 near-tie margin, overclaim guard). Provenance resists spoofing: near-exact quote matching rejects stitched quotes, and plans are grounded in the exact supplied passage set or declared empty. Structural role separation renders Heuer's disciplines as architecture <sup>[25](#reference-25)</sup>; the schema can represent contradiction and the fusion arithmetic is wired for it; parsing fails closed throughout.

**Weaknesses.** Confidence is an uncalibrated scalar that conflates quantity, quality and consistency, and an insufficient edge durably retains the rejected leading verdict's confidence. Causal content is inert prose: conditions and confounders are never written by any producer, and edge identity ignores mechanism, so contradictory mechanisms collapse into one edge. There is no ontology grounding, the failure mode that dominates SemRep's error budget and that services such as Gilda exist to fix <sup>[7](#reference-7), [46](#reference-46)</sup>. Trust tiers contribute nothing to confidence, so the evidence stream is bias-blind. Hardening is deferred: prompt-injection guards were removed from the graph-pipeline seams, and apply() skips all five gates for direct callers. Panel grades are unclamped, unknown IDs enter Borda tallies, there is no cycle detection, and batch-local evidence IDs invite mis-joins.

**The reachability gap is load-bearing.** The default workflow schedules only the support evidence role (`src/run_path.py`), and neither shipped entry point overrides it. Every committed evidence link therefore carries role support, the recorded counterevidence set is always empty, and any computed contradict, qualify or not_causal verdict trips the overclaim guard and is withheld. Each withholding is disclosed as an open risk, so nothing is hidden, but in every shipped run the five-verdict vocabulary collapses to support-or-abstain. Scheduling the contradiction, qualification, mechanism and confounder roles would activate already-implemented machinery for exactly the refutations the citation record under-supplies and Heuer's diagnosticity principle prizes. Generated hypotheses additionally need a post-expansion retrieval, scheduling and verification pass, because expansion runs after the workflow's only verification pass.

**Net.** The implementation is unusually transparent for its genre: its guards refuse the overclaims that literature-mining systems routinely commit, and its public docs and tests expose the remaining soft spots. Its two remaining limitations are epistemic range (the reachability gap) and epistemic depth (scalar, uncalibrated confidence over bias-uncorrected evidence). Both are addressable within the present representational choice, and the [Research Agenda](research-agenda.md) develops them as its first two studies.

## 4. Conditions for promoting a claim to a cause

"Impossible now" is accurate at literature scale, but it decomposes into seven conditions, ordered because each presupposes those below it.

1. **Well-defined variables.** Every node grounded to a stable ontology identity with measurement semantics (Gilda covers 1.8M names across nine ontologies with learned disambiguation <sup>[46](#reference-46)</sup>), defensible as a causal macro-variable <sup>[3](#reference-3)</sup>, and every cause given its intervention version <sup>[4](#reference-4)</sup>. Locally: ontology IDs, units and measurement fields on the concept node.
2. **Context-indexed edges.** Population, setting, dose and regime annotated, with mergers across contexts decided by transportability theory and selection diagrams <sup>[47](#reference-47)</sup>. In an exploratory analysis of SemMedDB predications, contextual differences explained most of the fifty-eight apparent contradictions that survived manual filtering <sup>[27](#reference-27)</sup>.
3. **A data plane.** Observational micro-data attached per context, so conditional-independence implications can be tested and some faithfulness violations diagnosed, while faithfulness stays partly untestable because the true graph is unknown <sup>[6](#reference-6)</sup>. This is the largest single addition: dataset ingestion, harmonisation and a statistics service.
4. **Explicit, bounded structural assumptions.** Causal sufficiency argued or relaxed through FCI-class methods that tolerate latent confounders at the price of weaker output <sup>[2](#reference-2)</sup>; acyclicity asserted or replaced by explicit feedback semantics; every assumption recorded as its own auditable claim.
5. **Claims demoted to priors.** Literature edges become soft priors and constraints for structure learning over pooled observational and experimental regimes, as in Joint Causal Inference <sup>[48](#reference-48)</sup>, with LLMs confined to the extractor and critic roles the reliability evidence supports <sup>[9](#reference-9), [12](#reference-12)</sup>.
6. **A closed intervention loop.** Experiments chosen by active intervention design <sup>[49](#reference-49)</sup>, executed, and fed back under POPPER-style sequential error control <sup>[23](#reference-23)</sup>. The plan objects exist; execution and feedback are the missing halves.
7. **Bias-corrected evidence semantics.** Publication- and citation-bias modelling <sup>[16](#reference-16), [17](#reference-17)</sup>, GRADE-style multidimensional certainty in place of one scalar <sup>[19](#reference-19)</sup>, and both difference-making and mechanistic evidence <sup>[21](#reference-21)</sup> before any edge is promoted.

**Architectural consequence: extend, do not convert.** Conditions 1–6 specify, evaluate and update a context-specific causal model; condition 7 is the project's additional evidential policy for promotion. The six may be jointly satisfiable in narrow regimes, such as one laboratory with a few harmonised variables and a Perturb-seq screen, where active-intervention methods already operate; for the open literature none is fully met. So keep the claim layer as assertion record, assumption ledger and prior; certify a derived causal layer only per context where 1–6 hold; require 7 before promoting an edge; and record each promotion as a transaction. This borrows the provenance-bearing assertion pattern of nanopublications <sup>[50](#reference-50)</sup> and INDRA's semantics for belief: the probability that a statement was correctly extracted and supported, deliberately not the probability that the relation holds <sup>[28](#reference-28)</sup>.

A bounded first step short of any of this is a task contract that distinguishes reporting what a source says, predicting an outcome, estimating an intervention effect, and transporting an effect to a new population <sup>[47](#reference-47), [51](#reference-51)</sup>. Each stronger task names its variables, target population, data, assumptions and identification method, and unknown requirements stay unknown. Prediction under distribution shift and causal transport have different mathematical requirements and should not collapse into a "domain similarity" score.

## 5. Case study: one hypothesis from a production run

A run anchored in CALDERA <sup>[52](#reference-52)</sup>, a paper on low-rank, low-precision LLM compression, produced 25 nodes and 10 edges with no authored priority. The examined hypothesis is a two-edge chain: inherent low-rank structure → conditions → rank–precision error coupling → determines effectiveness of → low-rank, low-precision decomposition. The outer concepts quote-resolve to CALDERA; the central coupling construct carries an empty provenance list, so the graph records the load-bearing concept as the hypothesis's own invention. The critic panel graded both mechanism steps "vague" and the coupling term "stretched". Both edges remain unverified with null confidence, because expansion creates them after the only verification pass. The attached plan is a preregistered factorial rank×precision ablation with matched baselines, three metrics with predicted directions and an explicit falsification clause, yet its grounding status is no_relevant_methods and all five control factors are marked from_graph: False, a Boolean authored by the Experiment Designer and not independently validated.

Under causal-graph semantics the outcome would differ in four ways: measurable variables (a low-rank-structure statistic S, rank r, precision b, reconstruction error E, downstream performance Y) instead of concepts; the rank–precision interaction as a non-additive term in the structural equation for E rather than a node; the chain re-expressed as effect modification and mediation in do-notation; and a quantitative terminus, an interaction coefficient with uncertainty. do(r, b) is literally executable here, so the plan can generate difference-making evidence, but until it runs neither difference-making nor mechanistic evidence exists <sup>[21](#reference-21)</sup>.

The case shows what the design solves: invention-by-citation is structurally blocked <sup>[16](#reference-16)</sup>, fabricated grounding is refused, a vague conjecture became a falsifiable protocol, and four plans were recovered from the append-only log through validated transactions. It also shows what it does not: zero evidence links graph-wide with no channel for refuting evidence, moderation mis-typed as prose, no contribution from the graph to confounding control, and a loop open at both ends, so the falsification clause can never fire inside the system.

## 6. Related systems and limitations

INDRA and its EMMAA extension are the closest large-scale precedent and share the decision that belief measures extraction correctness rather than causal truth <sup>[28](#reference-28)</sup>. CauseNet shows the noise regime that literature-scale assembly must survive <sup>[8](#reference-8)</sup>; SciFact supplies the claim-verification task shape <sup>[53](#reference-53)</sup>; nanopublications supply the assertion pattern <sup>[50](#reference-50)</sup>; POPPER and Kosmos bracket the design from the validation and world-model sides <sup>[23](#reference-23), [24](#reference-24)</sup>.

Five limits apply. External figures were taken from sources as read and not independently reproduced. The code discussion reflects the public checkout, so the support-only schedule is a claim about shipped defaults, not about what the architecture permits. The "why claims, not causes" rationale is a synthesis of visible implementation choices, public docs and the cited literature. The case study is one hypothesis in a domain friendlier to interventionist reconstruction than most the pipeline may serve. Whether the design yields better or more precise hypotheses awaits the `fhq-v1` results.

## References
<a id="reference-1"></a>
1. Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.
<a id="reference-2"></a>
2. Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search (2nd ed.). MIT Press.
<a id="reference-3"></a>
3. Beckers, S., & Halpern, J. Y. (2019). Abstracting causal models. Proceedings of the 33rd AAAI Conference on Artificial Intelligence.
<a id="reference-4"></a>
4. Hernán, M. A., & Robins, J. M. (2020). Causal Inference: What If. Chapman & Hall/CRC.
<a id="reference-5"></a>
5. Rojas-Saunero, L. P., et al. (2025). Inconsistent consistency: evaluating the well-defined intervention assumption in applied epidemiological research. International Journal of Epidemiology, 54(2), dyaf015.
<a id="reference-6"></a>
6. Zhang, J. (2016). The three faces of faithfulness. Synthese, 193, 1011–1027.
<a id="reference-7"></a>
7. Kilicoglu, H., Rosemblat, G., Fiszman, M., & Shin, D. (2020). Broad-coverage biomedical relation extraction with SemRep. BMC Bioinformatics, 21, 188.
<a id="reference-8"></a>
8. Heindorf, S., Scholten, Y., Wachsmuth, H., Ngonga Ngomo, A.-C., & Potthast, M. (2020). CauseNet: towards a causality graph extracted from the web. Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM).
<a id="reference-9"></a>
9. Kiciman, E., Ness, R., Sharma, A., & Tan, C. (2023). Causal reasoning and large language models: opening a new frontier for causality. Transactions on Machine Learning Research.
<a id="reference-10"></a>
10. Zecevic, M., Willig, M., Dhami, D. S., & Kersting, K. (2023). Causal parrots: large language models may talk causality but are not causal. Transactions on Machine Learning Research.
<a id="reference-11"></a>
11. Jin, Z., et al. (2024). Can large language models infer causation from correlation? Proceedings of the 12th International Conference on Learning Representations (ICLR).
<a id="reference-12"></a>
12. Feng, T., Qu, L., Tandon, N., Li, Z., Kang, X., & Haffari, G. (2025). On the reliability of large language models for causal discovery. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL).
<a id="reference-13"></a>
13. Reisach, A. G., Seiler, C., & Weichwald, S. (2021). Beware of the simulated DAG! Causal discovery benchmarks may be easy to game. Advances in Neural Information Processing Systems (NeurIPS) 34.
<a id="reference-14"></a>
14. Nastl, V., & Hardt, M. (2024). Do causal predictors generalize better to new domains? Advances in Neural Information Processing Systems (NeurIPS) 37.
<a id="reference-15"></a>
15. Brouillard, P., Squires, C., Wahl, J., Kording, K., Sachs, K., Drouin, A., & Sridhar, D. (2025). The landscape of causal discovery data: grounding causal discovery in real-world applications. Proceedings of the 4th Conference on Causal Learning and Reasoning (CLeaR).
<a id="reference-16"></a>
16. Greenberg, S. A. (2009). How citation distortions create unfounded authority: analysis of a citation network. BMJ, 339, b2680.
<a id="reference-17"></a>
17. Duyx, B., Urlings, M. J. E., Swaen, G. M. H., Bouter, L. M., & Zeegers, M. P. (2017). Scientific citations favor positive results: a systematic review and meta-analysis. Journal of Clinical Epidemiology, 88, 92–101.
<a id="reference-18"></a>
18. Cumpston, M. S., Brennan, S. E., Ryan, R., & McKenzie, J. E. (2023). Synthesis methods other than meta-analysis were commonly used but seldom specified: survey of systematic reviews. Journal of Clinical Epidemiology, 156, 42–52. doi:10.1016/j.jclinepi.2023.02.003.
<a id="reference-19"></a>
19. Guyatt, G. H., et al. (2008). GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. BMJ, 336, 924–926.
<a id="reference-20"></a>
20. Cartwright, N. (1989). Nature's Capacities and Their Measurement. Clarendon Press. (Ch. "No Causes In, No Causes Out.")
<a id="reference-21"></a>
21. Russo, F., & Williamson, J. (2007). Interpreting causality in the health sciences. International Studies in the Philosophy of Science, 21(2), 157–170.
<a id="reference-22"></a>
22. Min, S., et al. (2023). FActScore: fine-grained atomic evaluation of factual precision in long form text generation. Proceedings of EMNLP 2023.
<a id="reference-23"></a>
23. Huang, K., et al. (2025). [Automated hypothesis validation with agentic sequential falsifications (POPPER)](https://arxiv.org/abs/2502.09858). Proceedings of the 42nd International Conference on Machine Learning (ICML). arXiv:2502.09858.
<a id="reference-24"></a>
24. Mitchener, L., Yiu, E., Chang, S., et al. (2025). [Kosmos: an AI scientist for autonomous discovery](https://arxiv.org/abs/2511.02824). arXiv:2511.02824.
<a id="reference-25"></a>
25. Heuer, R. J., Jr. (1999). Psychology of Intelligence Analysis. Center for the Study of Intelligence, Central Intelligence Agency. (Analysis of Competing Hypotheses.)
<a id="reference-26"></a>
26. Hainy, M., Price, D. J., Restif, O., & Drovandi, C. (2022). Optimal Bayesian design for model discrimination via classification. Statistics and Computing, 32, 25.
<a id="reference-27"></a>
27. Rosemblat, G., Fiszman, M., Shin, D., & Kilicoglu, H. (2019). Towards a characterization of apparent contradictions in the biomedical literature using context analysis. Journal of Biomedical Informatics, 98, 103275.
<a id="reference-28"></a>
28. Gyori, B. M., Bachman, J. A., Subramanian, K., Muhlich, J. L., Galescu, L., & Sorger, P. K. (2017). From word models to executable models of signaling networks using automated assembly. Molecular Systems Biology, 13(11), 954.
<a id="reference-29"></a>
29. Yang, Z., Liu, W., Gao, B., Xie, T., Li, Y., Ouyang, W., Poria, S., Cambria, E., & Zhou, D. (2025). MOOSE-Chem: Large language models for rediscovering unseen chemistry scientific hypotheses. Proceedings of the International Conference on Learning Representations (ICLR). arXiv:2410.07076.
<a id="reference-30"></a>
30. Liu, H., Huang, S., Hu, J., Zhou, Y., & Tan, C. (2025). HypoBench: Towards systematic and principled benchmarking for hypothesis generation. arXiv:2504.11524.
<a id="reference-31"></a>
31. Xiong, G., Xie, E., Williams, C., Kim, M., Shariatmadari, A. H., Guo, S., Bekiranov, S., & Zhang, A. (2025). Toward reliable scientific hypothesis generation: Evaluating truthfulness and hallucination in large language models. Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI). arXiv:2505.14599.
<a id="reference-32"></a>
32. Gottweis, J., et al. (2026). [Accelerating scientific discovery with Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y). Nature, 655, 487–496. arXiv:2502.18864.
<a id="reference-33"></a>
33. Kambhampati, S., Valmeekam, K., Guan, L., Verma, M., Stechly, K., Bhambri, S., Saldyt, L., & Murthy, A. (2024). LLMs can't plan, but can help planning in LLM-Modulo frameworks. ICML 2024. arXiv:2402.01817.
<a id="reference-34"></a>
34. Cemri, M., et al. (2025). [Why do multi-agent LLM systems fail?](https://arxiv.org/abs/2503.13657) NeurIPS 2025, Datasets and Benchmarks Track. arXiv:2503.13657.
<a id="reference-35"></a>
35. Woodward, J. (2003). Making Things Happen: A Theory of Causal Explanation. Oxford University Press.
<a id="reference-36"></a>
36. Higgins, J. P. T., Thomas, J., Chandler, J., et al. (eds.) (2024). Cochrane Handbook for Systematic Reviews of Interventions, version 6.5. Cochrane. www.cochrane.org/handbook
<a id="reference-37"></a>
37. Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large language models cannot self-correct reasoning yet. ICLR 2024. arXiv:2310.01798.
<a id="reference-38"></a>
38. Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM evaluators recognize and favor their own generations. NeurIPS 2024. arXiv:2404.13076.
<a id="reference-39"></a>
39. Scannell, J. W., Blanckley, A., Boldon, H., & Warrington, B. (2012). Diagnosing the decline in pharmaceutical R&D efficiency. Nature Reviews Drug Discovery, 11(3), 191–200. doi:10.1038/nrd3681.
<a id="reference-40"></a>
40. Chow, C. K. (1970). On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory, 16(1), 41–46. doi:10.1109/TIT.1970.1054406.
<a id="reference-41"></a>
41. Kalai, A. T., Nachum, O., Vempala, S. S., & Zhang, E. (2025). Why language models hallucinate. arXiv:2509.04664.
<a id="reference-42"></a>
42. Messeri, L., & Crockett, M. J. (2024). Artificial intelligence and illusions of understanding in scientific research. Nature, 627(8002), 49–58. doi:10.1038/s41586-024-07146-0.
<a id="reference-43"></a>
43. Si, C., Yang, D., & Hashimoto, T. (2025). [Can LLMs generate novel research ideas? A large-scale human study with 100+ NLP researchers](https://arxiv.org/abs/2409.04109). ICLR 2025. arXiv:2409.04109.
<a id="reference-44"></a>
44. Weeber, M., Klein, H., de Jong-van den Berg, L. T. W., & Vos, R. (2001). Using concepts in literature-based discovery: Simulating Swanson's Raynaud–fish oil and migraine–magnesium discoveries. Journal of the American Society for Information Science and Technology, 52(7), 548–557. doi:10.1002/asi.1104.
<a id="reference-45"></a>
45. Jüni, P., Witschi, A., Bloch, R., & Egger, M. (1999). The hazards of scoring the quality of clinical trials for meta-analysis. JAMA, 282(11), 1054–1060. doi:10.1001/jama.282.11.1054.
<a id="reference-46"></a>
46. Gyori, B. M., Hoyt, C. T., & Steppi, A. (2022). Gilda: biomedical entity text normalization with machine-learned disambiguation as a service. Bioinformatics Advances, 2(1), vbac034.
<a id="reference-47"></a>
47. Bareinboim, E., & Pearl, J. (2016). [Causal inference and the data-fusion problem](https://ftp.cs.ucla.edu/pub/stat_ser/r450-corrected-proof.pdf). Proceedings of the National Academy of Sciences, 113(27), 7345–7352.
<a id="reference-48"></a>
48. Mooij, J. M., Magliacane, S., & Claassen, T. (2020). Joint causal inference from multiple contexts. Journal of Machine Learning Research, 21(99), 1–108.
<a id="reference-49"></a>
49. Zhang, J., Cammarata, L., Squires, C., Sapsis, T. P., & Uhler, C. (2023). Active learning for optimal intervention design in causal models. Nature Machine Intelligence, 5, 1066–1075.
<a id="reference-50"></a>
50. Groth, P., Gibson, A., & Velterop, J. (2010). The anatomy of a nanopublication. Information Services & Use, 30(1–2), 51–56.
<a id="reference-51"></a>
51. Lelova, Cooper, & Triantafillou (2026). [Transportability Without Graphs: A Bayesian Approach to Identifying s-Admissible Backdoor Sets](https://proceedings.mlr.press/v300/lelova26a.html). AISTATS, Proceedings of Machine Learning Research 300.
<a id="reference-52"></a>
52. Saha, R., Sagan, N., Srivastava, V., Goldsmith, A. J., & Pilanci, M. (2024). Compressing large language models using low rank and low precision decomposition (CALDERA). Advances in Neural Information Processing Systems (NeurIPS) 37.
<a id="reference-53"></a>
53. Wadden, D., et al. (2020). [Fact or fiction: verifying scientific claims](https://aclanthology.org/2020.emnlp-main.609/). Proceedings of EMNLP 2020.
