# From Traceable Claims to Testable Hypotheses: A Methodological Analysis of GraphHypothesize

*Complete report. A four-page decision version is at [traceable-claims-to-testable-hypotheses-decision.md](traceable-claims-to-testable-hypotheses-decision.md).*

## Abstract

Language-model systems can generate research hypotheses, relate them to literature and propose experiments. Assessing such systems requires distinguishing the integrity of the reasoning process from the scientific validity of its conclusions. This report examines that distinction through GraphHypothesize, a pipeline that represents relationships between concepts in a directed claim graph and governs changes through deterministic rules. Its principal contribution is an inspectable separation between model-generated assessments and the decisions that enter the research record. That separation does not establish the accuracy of extracted claims, the calibration of evidence assessments, the novelty of proposed hypotheses, or the informativeness of experiment plans. Analysis of the public implementation, representative output artifacts and relevant primary literature identifies limitations in claim semantics, access to counterevidence, belief revision, hypothesis evaluation and the completeness of the audit record. A descriptive comparison with scientific-discovery systems locates these limitations without treating different forms of validation as interchangeable. The resulting research agenda concerns whether traceability can support more reliable appraisal and revision of hypotheses. The available evidence supports claims about process control and artifact production; scientific performance remains to be established through independent evaluation.

**Keywords:** hypothesis generation; language models; claim graphs; scientific reasoning; evidence appraisal; provenance.

## 1. Introduction

Hypothesis generation connects existing knowledge to questions that can be investigated. Language models offer a way to assist this process by extracting concepts from literature, proposing relationships and specifying possible tests. Systems such as ResearchAgent and SciAgents develop literature-based ideas <sup>[1](#reference-1), [2](#reference-2)</sup>, while POPPER and Coscientist connect proposed investigations to data analysis or physical experiments <sup>[3](#reference-3), [4](#reference-4)</sup>. These activities produce different kinds of evidence about a system: a plausible proposal, an executable analysis and an experimentally supported finding are distinct achievements.

GraphHypothesize addresses the organisation and control of intermediate research claims. Concepts and their stated relationships are recorded in a graph, evidence assessments are attached to selected relationships, and proposed hypotheses pass through critique and confirmation before entering experiment planning. Deterministic rules govern which changes are accepted. This design makes the passage from a model response to a recorded decision available for inspection.

The research question is what this procedural discipline contributes to scientific reasoning. A system may preserve the origin of an assertion while misrepresenting its source. It may apply a confidence rule consistently while producing poorly calibrated assessments. It may record a detailed experiment plan without establishing that the experiment distinguishes the proposed explanation from alternatives. These distinctions motivate the central argument of this report: GraphHypothesize provides mechanisms for controlling a research record, but the scientific value of the record depends on additional representational and empirical conditions.

The analysis develops this argument in three steps. First, it distinguishes the objects and decisions that pass through the pipeline. Second, it examines how representation, evidence selection and hypothesis evaluation constrain the conclusions the system can support. Third, it identifies evaluations that could connect the architecture's process guarantees to measured research utility. The contribution is a methodological assessment and a set of testable research questions, not a report of newly measured system performance. The companion [Claims Before Causes](claims-before-causes.md) explains the representational choice itself; the [Research Agenda](research-agenda.md) develops the evaluations proposed here.

## 2. Scope and evidential basis

The analysis draws on the public implementation, representative exported artifacts, and primary literature relevant to the mechanisms discussed. Code inspection establishes what the workflow represents and permits. Output artifacts establish which records a run can produce. External studies supply evidence about related methods and motivate evaluation choices; their results are not measurements of GraphHypothesize.

Source selection for this report follows the questions addressed: preservation of claim meaning, retrieval of relevant and opposing evidence, reliability of model-based judgment, experimental testing, and maintenance of provenance. The comparison uses documented activities of contemporary discovery systems to distinguish forms of research support. This is a focused narrative analysis, not a systematic review or a comparative benchmark.

Conclusions about particular execution paths are scoped to the public workflow and shipped defaults. Implementation claims were checked against the code and relevant unit tests; those checks support the stated scope and are not an independent scientific validation of the system. Numerical results quoted from external studies are the original studies' reports and were not reproduced.

A representative exported run used in Sections 4.5 and 5 was parsed directly: 25 nodes, 10 edges and five plans. All ten edges were `unverified` with no edge-verification evidence links; three plans were marked `grounded` and two `no_relevant_methods`; the audit memo records zero refinement rounds. These are artifact counts and stored labels, not accuracy measurements. The run supplies a concrete counterexample to conflating plan commitment with hypothesis verification or experimental success; it was not randomly sampled and cannot estimate a failure rate.

## 3. The research objects and their transitions

GraphHypothesize represents concepts as nodes and relationships between them as directed edges. The graph is structured at the software level, but relationship labels remain free-form. An edge therefore records an assertion or a proposed relationship. Its presence does not establish a causal effect, and a graph of these assertions does not by itself supply an identified causal model.

In the claim-seeded workflow, an agent extracts concepts and relationships from an input claim. Retrieved literature is scored and selected for the relationships scheduled for review. An agent then produces evidential sub-signals, which deterministic rules map to a proposed verdict and confidence score. Further checks determine whether that verdict can be recorded. The final decision is therefore rule-governed, while its evidential inputs still depend on model judgment.

Hypothesis development uses graph structure and literature to propose candidates. Model-based critique and deterministic filtering and ranking determine which candidates are presented for confirmation. The graph view supplied to the hypothesis generator omits the recorded edge verdicts and confidence values. The design therefore should not be described as a demonstrated feedback loop in which verified relationships guide subsequent hypothesis generation. A confirmation gate selects candidates for commitment; this constitutes human oversight only when a person makes the selection, and an automated confirmation policy provides a different form of control.

Successfully committed hypotheses can enter experiment planning. Plans describe operationalisation, comparisons, measurements, expected outcomes and falsification criteria. They undergo required-content and citation-reference checks; model review and refinement depend on configuration. A plan may also explicitly record that no relevant methods support was found. These are research proposals whose completion and provenance can be inspected, rather than completed experimental tests.

**Table 1. Distinct meanings of validation in the pipeline.**

| Object | What an implemented check can establish | What requires additional evidence |
|---|---|---|
| Graph change | The proposal satisfies the applicable schema, reference, state-transition and duplicate-application rules | The represented claim preserves the intended scientific meaning |
| Evidence assessment | Preserved sub-signals are mapped to a verdict by specified rules | The verdict is accurate and its confidence is calibrated |
| Hypothesis candidate | The candidate meets configured criteria and was selected for commitment | It is novel, plausible and useful to investigate |
| Experiment plan | Required content is present and declared grounding IDs resolve within the available evidence scope | The cited passages support the plan, and the design is feasible and informative |
| Research record | Recorded sources and decisions can be inspected to the extent preserved | The complete run can be reconstructed and unauthorised changes detected |

## 4. Conditions for scientifically useful hypothesis generation

### 4.1 Preserving the meaning of a claim

The graph's identity and transaction mechanisms provide a stable basis for distinguishing recorded objects and inspecting accepted and rejected changes. Researcher-authored priority is separated from verification state, so preference for a topic cannot directly rewrite a relationship's verdict. These are meaningful controls over the production of the research record.

Their scientific interpretation depends on the representation inside that record. Free-form relationship labels can distinguish paraphrases that refer to the same relationship, so that "reduces" and "decreases" can create distinct edges, while broad labels can conceal differences in effect direction or scope. The schema includes textual conditions, confounders and scope information, but lacks dedicated fields for effect sign and magnitude and for a consistently structured population, intervention, comparator and outcome context. Opposing assertions may therefore reflect either a genuine disagreement or different experimental conditions. Detecting that distinction requires comparing their scope, not merely their endpoints. Code inspection also found that merging concepts could leave evidence or plan references attached to retired identities and could discard a prior verdict when edges collapsed.

Extraction introduces a related limitation. The examined claim-extraction path uses a single model response with structural parsing, while its auxiliary quality measures do not provide a general semantic admission check; a scope-gap calculation exists but, in the examined cycle, is computed after commitment when the extractor supplies the necessary facets. Initial extraction also does not use the concept-canonicalization procedure available during later enrichment. A structurally accepted graph can consequently omit a qualification or divide one concept across several records. Downstream appraisal then addresses the extracted assertion, which may differ from the assertion the researcher intended.

Claimify treats ambiguity, coverage and decontextualisation as explicit extraction concerns, and EDC separates extraction from schema definition and canonicalization <sup>[5](#reference-5), [6](#reference-6)</sup>. These studies motivate evaluating preservation of meaning and concept identity as distinct tasks; they do not establish that either method would improve GraphHypothesize without adaptation and testing. The relevant evaluation would ask whether researchers recover the same qualified proposition from the input and the extracted graph. It should separately examine omitted conditions, altered modality or negation, relation direction and duplicate concepts. Downstream agreement is insufficient when both the system and its evaluator are judging an incorrectly extracted claim.

### 4.2 Retrieving evidence that can change the assessment

Evidence retrieval has two responsibilities: identifying material relevant to the claim, and giving the appraisal process access to evidence that could change its conclusion. GraphHypothesize supports cross-source deduplication, recorded retrieval inputs and deterministic association scoring over a fixed SPECTER2 document representation <sup>[7](#reference-7)</sup>, with relevance signals combined by uncalibrated weights. These mechanisms aid inspection of the material selected for review. They do not establish retrieval completeness, stance coverage or independence among studies, and the selectivity of the relevance threshold needs measurement under both embedding and lexical retrieval.

The default scheduling policy requests support-role evidence, and every evidence role shares a claim-level pool obtained through topical search, with no explicit coverage estimate. The verdict architecture also represents contradiction, qualification and a non-causal interpretation, but recording those verdicts requires evidence carrying the corresponding roles. With only support-role links available, those disconfirming verdicts cannot pass the role requirement. This is a limitation of the default workflow, not of the verdict vocabulary. It constrains which conclusions can be recorded without demonstrating how frequently the system would make an incorrect supportive judgment. Edges for which no review is completed may remain unverified.

Merely adding more roles would not establish balanced evidence coverage. Topical similarity can retrieve papers about the same relationship without finding the strongest challenge to it, and deduplicating bibliographic records does not necessarily identify several papers that report the same underlying study. SciFact offers a methodological reference by separating retrieval of evidence-containing abstracts from support and refutation judgments and their rationales <sup>[8](#reference-8)</sup>; MultiVerS extends the task to full-document context under weak supervision <sup>[9](#reference-9)</sup>. Their labels provide an evaluation starting point rather than a complete model of causal appraisal.

A suitable assessment would measure relevant-evidence recall and stance coverage separately, using a bounded corpus for which reference evidence has been established. It would then test whether supplying previously omitted counterevidence changes the verdict appropriately. Failure to retrieve evidence should remain distinguishable from a model or backend failure and from a substantive absence of informative studies, and one-shot retrieval should be compared with adaptive retrieval at matched cost.

### 4.3 Mapping evidence to conclusions and revising them

Deterministic fusion makes the decision rule explicit: the same preserved inputs and configuration should produce the same computed assessment. An abstention outcome also allows the system to withhold a settled conclusion. These properties facilitate evaluation because researchers can locate whether a disagreement originates in the evidence, the model's sub-signals or the rule that combines them.

Neither property establishes calibration. The examined workflow uses configured weights and thresholds without a demonstrated relationship between reported confidence and observed correctness. Its confidence values should therefore be interpreted as outputs of the appraisal rule rather than as probabilities that a claim is true. Comparison with POPPER clarifies the boundary: POPPER designs and executes falsification tests within a sequential framework with Type-I error control under its testing assumptions <sup>[3](#reference-3)</sup>. Deterministic literature appraisal and statistical error control address different verification problems. Evaluation designs for claim verification (SciFact, FEVER, CheckWhy, LLM-AggreFact) would each need adaptation to the graph's claim types and verdict semantics <sup>[8](#reference-8), [11](#reference-11), [12](#reference-12), [13](#reference-13)</sup>, and work on calibrated escalation motivates measuring error among accepted verdicts rather than treating fixed arithmetic as a reliability guarantee <sup>[14](#reference-14)</sup>.

Null findings also require a more precise treatment than adding a single verdict label. At the study level, a null finding can be accompanied by an effect estimate, an uncertainty interval and information about test sensitivity. Its implication for a claim depends on the effect the study could have detected and the conditions investigated. It may contradict a sufficiently specific prediction, qualify its scope, or leave it unresolved. A useful extension would represent that evidence before deriving the claim-level conclusion.

The state-transition rules impose a further limitation: supported, contradicted, qualified and non-causal verdicts are terminal through the verification interface. An insufficient verdict can move to a substantive verdict, but a settled substantive verdict cannot be reopened by the same mechanism. This restricts response to new evidence or a corrected interpretation. Temporal knowledge-graph work such as Zep illustrates a different approach, maintaining validity intervals and invalidating earlier assertions while retaining their history <sup>[10](#reference-10)</sup>. It supplies an architectural example, not evidence of a field-wide consensus or a validated solution for scientific claims.

These observations motivate two linked evaluations: calibration on independently labelled claim–evidence sets, and revision under controlled additions or corrections to evidence. Preserving earlier decisions and revising current assessments are compatible research requirements. The architecture's inability to revise a settled verdict should not be treated as a necessary price of auditability.

### 4.4 Evaluating hypotheses beyond fluent novelty judgments

GraphHypothesize separates candidate generation, critique and commitment. The critic panel receives candidate structure and grounding material while excluding the generator's private rationale and self-grades. Its judgments are aggregated by deterministic rules. This separation makes the contributions of the generator, judges and selection policy more inspectable.

The scoring signals nevertheless require validation. Candidate eligibility combines model judgments with embedding-based comparisons, and most components of the subsequent composite ranking remain generator-provided assessments. The panel supplies a separate field-novelty judgment and a literature-saturation assessment. These signals serve different purposes: distance from existing concept descriptions is not the same construct as novelty of a relationship, and novelty is not equivalent to plausibility, testability or research value. A saturation penalty can also lower the rank of a worthwhile replication or boundary-condition study.

RINoBench reports substantial divergence between model novelty judgments and expert reference judgments in its evaluated settings; in its zero-shot conditions, models avoided the "not novel" category and achieved a best macro-F1 of 17.2 <sup>[15](#reference-15)</sup>. RQ-Bench similarly finds that model judges favour generated research questions while experts prefer author-anchored reference questions, identifies source-boundedness as a relevant failure mode, and reports that greater embedding distance did not align with expert novelty preferences in an evaluation of fifty computer-science research-question pairs <sup>[16](#reference-16)</sup>. These findings warrant testing GraphHypothesize's measures against expert judgments. They do not show that every model-based novelty score is invalid or establish an error rate for this panel.

Panel composition presents a separate question. PoLL provides evidence that diverse model panels can improve evaluation in the tasks it studies <sup>[17](#reference-17)</sup>, but another study found that nine judges supplied only about two independent votes' worth of information <sup>[18](#reference-18)</sup>. In GraphHypothesize, the reviewed configuration uses different models with prompt-level separation from proposer internals and median aggregation, but model diversity is not enforced, and additional review is triggered only by selected forms of disagreement (on the first-ranked candidate or on two categorical flags). Agreement among judges cannot reveal a mistake they share. Candidate order is shared across judges, which leaves candidate identity confounded with presentation position without establishing perfectly correlated position bias, and numerical grades lack range checks. Taking the median of two grades is equivalent to taking their mean, so the base panel cannot inherit every robustness property associated with larger median-aggregated panels. Self-preference and limits of self-correction warrant further evaluation <sup>[19](#reference-19), [20](#reference-20)</sup>, and position and score-range biases provide additional reasons to validate the panel <sup>[21](#reference-21), [22](#reference-22)</sup>. In the novelty-judge study, the original prompt produced 52% judge–judge agreement and expert–judge agreement as low as 22%; a revised condition produced 62% and 30–40% respectively <sup>[16](#reference-16)</sup>.

Evaluation should distinguish novelty of concepts, relationships, mechanisms and applications; include replication value; and ask experts to assess candidates without knowing their origin or model scores. Counterbalanced presentation and assessment of the full candidate pool would separate judgment quality from ordering effects and selection losses, and expert-anchored meta-evaluation is available as a design <sup>[23](#reference-23)</sup>. Tournament ranking, used by Co-Scientist and Robin, is a candidate intervention, not an established remedy: changing the ranking procedure cannot by itself supply a valid scientific criterion.

### 4.5 Turning a hypothesis into an informative test

Experiment planning translates a proposed relationship into an intervention or comparison, measurements, and possible observations that would count against it. GraphHypothesize makes several of these elements explicit and checks required content before commitment. Declared grounding IDs must resolve within the relevant retrieved material, which prevents references to nonexistent evidence, and plans can explicitly record the absence of relevant methods evidence. This provides useful distinctions among a cited proposal, an uncited proposal and an incomplete response.

Reference resolution establishes only that the declared source record is available. The plan-admission checks do not establish that a supplied quotation occurs verbatim in that source or that it justifies the proposed method. Model-based plan review does not impose a numerical acceptance threshold, a final refinement is not necessarily reviewed again, and the required-content list does not require the procedure field. Consequently, structural acceptance and model review should not be described as scientific validation.

The remaining question is whether a design would discriminate between competing explanations. A detailed plan can be feasible yet weakly informative if its predicted observations are also expected under plausible alternatives. BED-LLM motivates an explicit information-gain objective, but its demonstrated tasks concern question selection and preference elicitation, so transfer to scientific experiment design remains a research proposal <sup>[24](#reference-24)</sup>. Deterministic mediation of scientific workflows and protocol-planning evaluation supply relevant design precedents <sup>[25](#reference-25), [26](#reference-26)</sup>. Kosmos's expert audit validated 82.1% of its literature-based statements through independent source searches; that result concerns statement accuracy and is not directly comparable with evidence-ID validation <sup>[27](#reference-27)</sup>.

The preserved execution artifacts reinforce the distinction. The run described in Section 2 contains committed experiment plans alongside unverified hypothesis edges, including plans explicitly marked as having no relevant methods support. It demonstrates production and recording of research proposals. Those records do not supply experimental outcomes or independent scientific judgments validating the proposals.

An evaluation of experimental value would ask independent researchers whether a plan is executable, whether the claimed source supports its method, and which alternative explanations its results could distinguish. Subsequent execution would supply a different and stronger form of evidence. A hypothesis need not be confirmed for its experiment to be useful: an informative negative result can improve the research record if the system can incorporate it.

### 4.6 Maintaining an inspectable and revisable research record

The transaction core assigns content-based identities, checks changes before acceptance, and records accepted and rejected proposals. These mechanisms provide a basis for tracing decisions. Strong claims about complete replay and tamper evidence require qualification. The examined graph-state path records events without linking them through predecessor hashes, and the ordinary exported receipts do not contain the complete transaction payloads required by the replay routine; a replay routine exists and passes its unit test, but complete replay from the ordinary exports is a different claim. Exported retrieval material and visible traces are valuable, but they do not alone demonstrate complete reconstruction of graph history or detection of unauthorised changes. Provenance is not exported in PROV- or nanopublication-compatible form.

Record integrity also depends on maintaining the meaning of references across transformations. The examined node-merge operation recomputes edge identities without remapping associated evidence links and experiment plans. This can compromise later reconstruction of which evidence or plan belonged to a relationship; it is described here as a possible continuity failure, not a measured rate of corruption. A complete audit has to test semantic continuity as well as the presence of hashes and logs.

Retrieved content introduces an additional vulnerability because it reaches model-based appraisal. The examined graph-pipeline prompts lack the explicit instruction–data framing present in the web-retrieval prompts. This supports a concern about the boundary between evidence and instructions; it does not establish a measured attack rate. Indirect prompt injection has been demonstrated against model-based scientific reviewers <sup>[29](#reference-29)</sup>, and CaMeL shows how control flow and untrusted data can be separated architecturally, underscoring that a prompt warning is not an enforced security boundary <sup>[28](#reference-28)</sup>. Hash-linked logging has been proposed for agent audit trails, though only as an individual Internet-Draft rather than an established standard <sup>[30](#reference-30)</sup>, and provenance frameworks supply options for interoperable research records <sup>[31](#reference-31), [32](#reference-32)</sup>.

For research use, the practical objective is an account that another investigator can reconstruct and challenge: which claim was assessed, which source passage was available, how the assessment was derived, and what later evidence changed it. Completing provenance and replay mechanisms would support that objective. It would make erroneous reasoning more inspectable without making the reasoning correct by construction.

## 5. Position among scientific-discovery systems

The comparison is organised by documented research activity and form of evaluation. It assigns no overall ranking: literature appraisal, idea assessment, data analysis and laboratory testing lack a common outcome measure in the evidence considered here. A capability omitted from a paper is not assumed to be absent from its implementation.

**Table 2. Research activities and the evidence needed to interpret them.**

| System | Documented focus | Evidential distinction relevant to this analysis |
|---|---|---|
| GraphHypothesize | Controlled changes to a claim graph; evidence appraisal; hypothesis critique; experiment proposals | Implementation and saved outputs establish process behaviour; scientific accuracy remains unmeasured in the evidence examined |
| ResearchAgent <sup>[1](#reference-1)</sup> | Literature- and entity-informed idea generation with iterative reviewing agents | Human and model evaluations concern idea quality; these differ from executing the proposed research |
| SciAgents <sup>[2](#reference-2)</sup> | Ontological graph reasoning and multi-agent hypothesis development | Graph-linked case studies illustrate ideation; experimental support for a particular proposal requires separate evidence |
| The AI Scientist <sup>[33](#reference-33)</sup> | Idea generation, code, computational experiments, manuscripts and automated review | Computational execution and manuscript scoring address different criteria; successful execution does not establish correctness of every reported interpretation |
| Kosmos <sup>[27](#reference-27)</sup> | Iterative data analysis and literature research coordinated through a world model | Its reported expert statement audit measures output accuracy, a form of evidence distinct from process checks |
| Co-Scientist <sup>[35](#reference-35)</sup> | Iterative generation, critique and tournament evolution of research hypotheses | Biomedical validations support selected applications; they do not calibrate every generated hypothesis or ranking |
| POPPER <sup>[3](#reference-3)</sup> | Executed falsification tests with sequential statistical control | Type-I error guarantees concern the testing framework and its assumptions, rather than general accuracy of literature synthesis |
| Robin <sup>[38](#reference-38)</sup> | Hypothesis generation and data analysis connected to laboratory feedback | The reported discovery process is semi-autonomous and coordinates with scientists who perform the experiments |
| Coscientist <sup>[4](#reference-4)</sup> | Tool-assisted planning and execution of chemistry experiments | Physical execution demonstrates capabilities beyond plan production, with evaluation bounded by the reported chemistry tasks |

The comparison suggests complementary research objectives rather than a single ordering of systems. GraphHypothesize emphasises explicit claim state and inspectable decisions. POPPER addresses sequential statistical verification, while Robin and Coscientist connect proposals to experimental execution. These capabilities require separate assessment: deterministic appraisal and statistical error control are not equivalent forms of verification.

The external record supplies reference points, not measurements of this system. Kosmos reports 79.4% accuracy across 102 statements from three reports, falling to 57.9% for synthesis statements <sup>[27](#reference-27)</sup>. Robin demonstrates semi-autonomous laboratory discovery in which scientists reviewed candidates, developed or modified protocols and executed experiments <sup>[38](#reference-38)</sup>. Co-Scientist reports scientist participation and biological validations <sup>[35](#reference-35), [36](#reference-36)</sup>; an independent reimplementation report raises reproducibility questions, though failure to reproduce particular hypotheses would not by itself invalidate the original experimental findings <sup>[37](#reference-37)</sup>. POPPER's demonstrated instantiation operates over a preloaded data corpus <sup>[3](#reference-3)</sup>. Coscientist demonstrates execution of established chemical reactions <sup>[4](#reference-4)</sup>, while an independent study questions the benefit of experimental feedback in the evaluated optimisation settings <sup>[39](#reference-39)</sup>. An evaluation of the AI Scientist reports failed experiments and hallucinated manuscript numbers <sup>[34](#reference-34)</sup>. SciAgents' cited study does not establish experimentally validated hypotheses <sup>[2](#reference-2)</sup>. HindSight reports a negative association between ResearchAgent's model-judged novelty and its retrospective measure of subsequent research; that proxy should not be equated with scientific value generally <sup>[43](#reference-43)</sup>.

These results emphasise the need to distinguish structural conformance from output validity. SciEx reported precision of at most 0.333 in its evaluated extraction settings, and did not report perfect schema validity <sup>[40](#reference-40)</sup>. Across more than 25,000 scientific-agent runs, one study reports that traces ignored available evidence 68% of the time and revised beliefs only 26%, with the base model explaining 41.4% of outcome variance versus 1.5% for the scaffold <sup>[41](#reference-41)</sup>. A separate position paper argues more broadly that current agent architectures are poorly suited to autonomous discovery <sup>[42](#reference-42)</sup>. These findings motivate direct evaluation of GraphHypothesize; they do not measure its performance.

A testable research direction is to evaluate whether GraphHypothesize can serve as a persistent claim layer beneath an execution-capable system such as POPPER or Robin. Such a study would need an explicit mapping from proposed hypotheses to tested predictions, procedures, observations and revised assessments, and it would need to preserve disagreements and negative findings. Compatibility, added scientific value and integration cost remain to be established.

## 6. Evaluations the findings call for

The central research question is whether a persistent claim graph helps researchers reach and revise scientifically warranted conclusions. This requires more than showing that the pipeline produces orderly artifacts. An intervention may improve extraction without improving appraisal, improve model agreement without improving expert agreement, or improve protocol detail without making an experiment more informative. The agenda therefore treats these transitions as separately testable mechanisms before assessing the complete system. Recent literature sharpens the distinction: scientific claim-verification benchmarks make evidence selection and verdict prediction separable, research-idea studies distinguish promising generation from unreliable judging, and discovery benchmarks increasingly inspect executed analyses and the evidence behind a claim. These developments supply evaluation designs, not transferable performance estimates.

**Table 3. Proposed studies and their principal contrasts. None has been conducted.**

| Study | Hypothesis to test | Principal comparison | Primary outcome |
|---|---|---|---|
| Semantic fidelity | Explicit conditions and claim decomposition reduce meaning-changing errors | Current extraction versus scope-aware extraction with a common model and source set | Expert-adjudicated claim fidelity, including omissions |
| Exposure to counterevidence | Deliberate evidence-role coverage reduces unsupported acceptance | Support-oriented versus balanced retrieval at a matched budget | Counterevidence coverage and accepted-claim error |
| Calibrated acceptance | Empirical selection improves reliability at useful coverage | Existing score threshold versus held-out calibration and escalation | Disagreement risk among accepted cases, coverage and review cost |
| Warranted revision | An explicit update procedure improves response to new evidence without destabilising unaffected claims | Incremental revision versus full recomputation from the same evidence | Correct revision and preservation of unaffected conclusions |
| Hypothesis selection | Critique adds value beyond generator scores and similarity-based novelty | Selection procedures applied to the same candidate pool | Blinded expert preference and shared judge error |
| Informative experiments | Explicit alternatives improve discriminating value beyond plan completeness | Existing review versus alternative-aware review with matched resources | Discrimination between explanations and warranted outcome interpretation |
| Researcher utility | A graph improves investigation beyond access to the underlying evidence | Graph-assisted work versus an equally complete flat dossier | Correct evidence-based decisions, error localisation and investigator time |

**Semantic fidelity.** A corpus would pair source passages with independently annotated claims, including polarity, modality, relation direction, population or system, intervention, comparator, outcome and conditions where applicable, sampling qualified associations, causal assertions, null findings and statements whose conclusions change when a condition is removed. Countercausality research supplies a useful distinction, since causing an outcome to decrease and denying that a causal relation exists express different propositions; its news-text experiments motivate scientific-text probes rather than establishing scientific extraction accuracy <sup>[44](#reference-44)</sup>. The comparison would hold the model, passages and budget constant while varying decomposition and scope checking, with canonicalization tested separately, and experts judging both retained meaning and omitted content, because merely checking whether emitted edges are correct rewards a system that avoids difficult claims. CLAIM-BENCH extends this assessment to claims, evidence spans and their links within full papers, and its annotation difficulties motivate adjudication of the evidence links themselves <sup>[45](#reference-45)</sup>. A further comparison would supply expert-corrected claims to the unchanged downstream pipeline: if downstream verdicts improve substantially, extraction is an identifiable bottleneck.

**Exposure to counterevidence.** A curated evidence collection would distinguish support, contradiction, scope restriction, methodological criticism and irrelevant material, with additional annotations for explicit denial of causation, evidence consistent with no meaningful effect, and inconclusive estimates. CONFACT motivates testing conflicting documents together with source metadata, though media-source credibility cannot substitute for scientific study appraisal <sup>[46](#reference-46)</sup>. Evidence should be grouped by underlying study, since multiple articles based on one dataset must not automatically count as independent corroboration. The experiment would compare support-oriented retrieval with balanced evidence-role scheduling under the same claim, query budget and reading budget, holding the appraiser constant to isolate retrieval and supplying the same expert-curated evidence to every appraiser to isolate appraisal. Generated queries need a separate audit, because query-expansion research finds that some apparent retrieval gains coincide with generated text already containing information supported by the reference evidence <sup>[47](#reference-47)</sup>. The intervention would fail if additional counterevidence queries increased document volume without improving role coverage or reducing unsupported acceptance; if counterevidence is recovered but ignored, the result identifies an appraisal or commitment problem instead.

**Calibrated acceptance.** Trust or Escalate offers the selective-evaluation design, with a statistical target of agreement with human preferences under stated calibration assumptions <sup>[14](#reference-14)</sup>. The target for GraphHypothesize must be defined before fitting a calibration model; one defensible target is agreement with independent expert adjudication of whether a claim is supported by the supplied evidence, which differs from the probability that the claim is true. Existing deterministic scores should first be evaluated as scores; any mapping to probabilities should be fitted on development data and assessed on a separate test set with paper families in one partition. The main comparison holds candidate judgments fixed while varying the acceptance policy, reporting error among accepted judgments, the fraction accepted, a risk–coverage curve and the cost of escalation. A useful negative result would be failure to preserve reliability outside the calibration domain, or no improvement over a simpler threshold at matched coverage and cost.

**Warranted revision.** DeReLab provides formally generated defeasible-reasoning updates, including exceptions and irrelevant information, and BayesBench distinguishes inference about hidden structure from the downstream predictions that should use it <sup>[48](#reference-48), [49](#reference-49)</sup>; their reference models do not determine the correct interpretation of heterogeneous scientific studies. Evidence sequences with independently specified desired updates would include genuine contradiction, a narrower population, duplicated evidence, withdrawal of a supporting source and irrelevant information, with confirming and disconfirming updates matched for relevance and strength. The stronger comparison is between an incremental procedure with preserved history and full recomputation from the same accumulated evidence, with a second contrast varying evidence order. Outcomes are whether each required revision occurs, whether unaffected claims remain stable, whether qualified conclusions replace overbroad ones, and whether downstream hypotheses and plans are reconsidered; turn-by-turn assessment is needed because stable final accuracy can conceal earlier failures.

**Hypothesis selection.** In a blinded human study, Si and colleagues found promising novelty ratings for AI-generated NLP research ideas while also finding limitations in model-based ranking <sup>[50](#reference-50)</sup>. For sampled research problems, the complete generated candidate pool would be retained, and experts would assess candidates with the same relevant source context while model scores, selection status and generator identity are hidden, separating relationship novelty, mechanistic plausibility, testability, feasibility and replication value. Selection procedures would then operate on that same pool: generator-provided scores, similarity-based eligibility, the current critic procedure and budget-matched heterogeneous judging, with pairwise comparisons evaluated in both presentation orders. SciMON provides a nearby literature- and graph-informed generation comparator <sup>[51](#reference-51)</sup>, and the Geometry of LLM-as-Judge shows why inter-model consensus requires an independent human reference <sup>[52](#reference-52)</sup>. A randomly selected audit of cases where the base judges agree is essential. A temporal evaluation against later publications can supplement expert appraisal but does not measure scientific truth, and a frozen retrieval cutoff does not remove later knowledge already present in model weights <sup>[43](#reference-43)</sup>.

**Informative experiments.** The intervention would require explicit alternatives, predicted outcome patterns under each explanation, an observable contrast, and rules specifying which outcomes maintain, narrow or weaken the claim, compared with the existing review under matched budgets and plan length. Controlled computational environments should precede claims of general experimental value, comparing prespecified design choices against simple fixed designs and, where tractable, known-optimal references, and reporting realised discrimination separately from estimated expected information gain. Any POPPER-style execution layer must examine whether each generated implication is logically appropriate and whether the executed tests satisfy the statistical assumptions; a guarantee for one testing sequence should not be presented as control over the entire discovery program <sup>[3](#reference-3)</sup>. Generation, eligibility, execution and interpretation require separate denominators, retaining unsuccessful executions and null outcomes; ReplicatorBench provides a replication-workflow precedent <sup>[53](#reference-53)</sup>.

**Researcher utility.** Researchers could be assigned matched investigation tasks with a final report alone, an equally complete flat evidence dossier, or the dossier plus the claim graph and revision history. Tasks should require locating support for a claim, identifying a scope mismatch, distinguishing independent studies from repeated reports, explaining a revision, and determining which downstream proposals a correction affects. Reconstruction needs distinct tests: replaying preserved transactions to recover a state differs from rerunning model calls, from computationally reproducing an analysis, and from replicating a scientific result with new data. End-to-end evaluation should retain this decomposition; DiscoveryBench, ScienceAgentBench and TruthInsightBench are complementary task designs whose model-based grading does not justify treating any rubric as an established measure of discovery quality <sup>[54](#reference-54), [55](#reference-55), [56](#reference-56)</sup>. The strongest support for the architecture would be an improvement over the equally complete dossier in warranted decisions or efficient correction, sustained when evidence changes.

The proposed order is to establish semantic and appraisal measurement first, then test revision, selection and experiment design, and finally conduct a prospective researcher study. The [Research Agenda](research-agenda.md) develops four of these studies in full, as a chain from counterevidence through calibration and revision to researcher benefit, and treats semantic fidelity, citation warrant and provenance as foundations.

## 7. Limitations

This report establishes no new performance estimates. The implementation analysis is focused rather than exhaustive, and some properties vary with configuration or execution path. The representative run clarifies the distinction between generated artifacts and scientific validation; it is not representative and cannot estimate accuracy or failure prevalence. No model or scientific experiment was rerun for this assessment. The literature comparison is limited by differences in tasks, domains, evaluation criteria and reporting; results from external studies motivate concerns and possible interventions, but do not determine how frequently the corresponding failures occur in GraphHypothesize. Comparator evidence varies between peer-reviewed studies, preprints, vendor reports and independent reimplementations. Source selection is explicit and question-driven, but not systematic or exhaustive. Accordingly, the report makes no claim of field-wide consensus, complete coverage, or superiority over the compared systems.

## 8. Conclusion

GraphHypothesize provides an inspectable framework for representing claims, appraising evidence, developing hypotheses and recording experiment proposals. Its principal contribution lies in the control and traceability of those transitions. The scientific usefulness of the resulting record depends on preserving claim meaning, admitting counterevidence, calibrating judgments, assessing hypotheses against appropriate criteria, and incorporating the outcomes of informative tests. The available evidence therefore supports a bounded conclusion: process integrity is a useful foundation for hypothesis-generation research, while scientific correctness and research utility remain empirical questions. Completing the audit record and enabling supervised revision would make those questions easier to investigate. Their answers require independent evaluation of the claims, hypotheses and experiments the system produces.

## References
<a id="reference-1"></a>
1. Baek, J., Jauhar, S. K., Cucerzan, S., & Hwang, S. J. (2025). [ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models](https://arxiv.org/abs/2404.07738). NAACL.
<a id="reference-2"></a>
2. Ghafarollahi, A., & Buehler, M. J. (2024). [SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](https://arxiv.org/abs/2409.05556). arXiv:2409.05556; Advanced Materials (2025).
<a id="reference-3"></a>
3. Huang, K., Jin, Y., Li, R., Li, M. Y., Candès, E., & Leskovec, J. (2025). [Automated Hypothesis Validation with Agentic Sequential Falsifications (POPPER)](https://arxiv.org/abs/2502.09858). ICML; arXiv:2502.09858.
<a id="reference-4"></a>
4. Boiko, D. A., MacKnight, R., Kline, B., & Gomes, G. (2023). [Autonomous chemical research with large language models](https://www.nature.com/articles/s41586-023-06792-0). Nature, 624, 570–578.
<a id="reference-5"></a>
5. Metropolitansky, D., & Larson, J. (2025). [Towards Effective Extraction and Evaluation of Factual Claims (Claimify)](https://arxiv.org/abs/2502.10855). ACL.
<a id="reference-6"></a>
6. Zhang, B., & Soh, H. (2024). [Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction (EDC)](https://aclanthology.org/2024.emnlp-main.548/). EMNLP, 9820–9836.
<a id="reference-7"></a>
7. Singh, A., et al. (2023). SciRepEval / SPECTER2: Scientific Document Representations.
<a id="reference-8"></a>
8. Wadden, D., et al. (2020). [Fact or Fiction: Verifying Scientific Claims (SciFact)](https://aclanthology.org/2020.emnlp-main.609/). EMNLP, 7534–7550.
<a id="reference-9"></a>
9. Wadden, D., et al. (2022). MultiVerS: Improving Scientific Claim Verification with Weak Supervision and Full-Document Context. Findings of NAACL.
<a id="reference-10"></a>
10. Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/html/2501.13956v1). arXiv:2501.13956.
<a id="reference-11"></a>
11. Thorne, J., et al. (2018). The Fact Extraction and VERification (FEVER) Shared Task.
<a id="reference-12"></a>
12. Si, J., et al. (2024). CheckWhy: Causal Fact Verification via Argument Structure. ACL.
<a id="reference-13"></a>
13. Tang, L., et al. (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents (LLM-AggreFact). EMNLP.
<a id="reference-14"></a>
14. Jung, J., Brahman, F., & Choi, Y. (2025). [Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](https://arxiv.org/abs/2407.18370). ICLR 2025; arXiv v1 first submitted 25 July 2024.
<a id="reference-15"></a>
15. Schopf, T., & Färber, M. (2026). [Is This Idea Novel? An Automated Benchmark for Judgment of Research Ideas (RINoBench)](https://arxiv.org/abs/2603.10303). arXiv:2603.10303; accepted to LREC 2026.
<a id="reference-16"></a>
16. Sinhahajari, S., Majumder, N., & Poria, S. (2026). [On the Limits of LLM-as-Judge for Scientific Novelty Assessment (RQ-Bench)](https://arxiv.org/abs/2606.12071). arXiv:2606.12071.
<a id="reference-17"></a>
17. Verga, P., et al. (2024). [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL)](https://arxiv.org/abs/2404.18796). arXiv:2404.18796.
<a id="reference-18"></a>
18. [Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels](https://arxiv.org/abs/2605.29800). arXiv:2605.29800 (2026).
<a id="reference-19"></a>
19. [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/abs/2410.21819). arXiv:2410.21819 (2024).
<a id="reference-20"></a>
20. Huang, J., et al. (2024). [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798). ICLR.
<a id="reference-21"></a>
21. Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS.
<a id="reference-22"></a>
22. [Score Range Bias in LLM-as-a-Judge](https://arxiv.org/abs/2510.18196). arXiv:2510.18196 (2025).
<a id="reference-23"></a>
23. Chern, S., Chern, E., Neubig, G., & Liu, P. (2024). [Can Large Language Models Be Trusted for Evaluation? Scalable Meta-Evaluation via Agent Debate (ScaleEval)](https://arxiv.org/abs/2401.16788). arXiv:2401.16788.
<a id="reference-24"></a>
24. Choudhury, D., et al. (2026). [BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design](https://arxiv.org/abs/2508.21184). ICLR.
<a id="reference-25"></a>
25. Adamidis, P., et al. (2026). [It's Not the Language Model, It's the Tool: Deterministic Mediation for Scientific Workflows](https://arxiv.org/abs/2605.13245). arXiv:2605.13245.
<a id="reference-26"></a>
26. O'Donoghue, O., et al. (2023). [BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology](https://arxiv.org/abs/2310.10632). EMNLP.
<a id="reference-27"></a>
27. Mitchener, L., et al. (2025). [Kosmos: An AI Scientist for Autonomous Discovery](https://arxiv.org/abs/2511.02824). arXiv:2511.02824.
<a id="reference-28"></a>
28. Debenedetti, E., et al. (2025). [Defeating Prompt Injections by Design (CaMeL)](https://arxiv.org/abs/2503.18813). arXiv:2503.18813.
<a id="reference-29"></a>
29. Sahoo, S., et al. (2025). [When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection](https://arxiv.org/abs/2512.10449). arXiv:2512.10449.
<a id="reference-30"></a>
30. Sharif, R. (2026). Agent Audit Trail: A Standard Logging Format for Autonomous AI Systems. Individual IETF Internet-Draft draft-sharif-agent-audit-trail-00, work in progress.
<a id="reference-31"></a>
31. Souza, R., et al. (2025). [PROV-AGENT: Unified Provenance for Tracking AI Agent Interactions in Agentic Workflows](https://arxiv.org/abs/2508.02866). IEEE eScience.
<a id="reference-32"></a>
32. Extending Nanopublications with Knowledge Provenance. CEUR-WS Vol. 3937 (2025).
<a id="reference-33"></a>
33. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292). arXiv:2408.06292; "Towards End-to-End Automation of AI Research," Nature (2026).
<a id="reference-34"></a>
34. Beel, J., Kan, M.-Y., & Baumgart, M. (2025). [Evaluating Sakana's AI Scientist for Autonomous Research: Wishful Thinking or an Emerging Reality?](https://arxiv.org/abs/2502.14297) arXiv:2502.14297; ACM SIGIR Forum.
<a id="reference-35"></a>
35. Gottweis, J., et al. (2026). [Accelerating scientific discovery with Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y). Nature, 655, 487–496.
<a id="reference-36"></a>
36. Penadés, J. R., Costa, T. R. D., et al. (2025). AI Mirrors Experimental Science to Uncover a Novel Mechanism of Gene Transfer. Cell; bioRxiv 2025.02.19.639094.
<a id="reference-37"></a>
37. Huang, K.-L. (2026). I Re-Implemented Google's AI Co-Scientist. Independent reproduction report, jrnlclub.
<a id="reference-38"></a>
38. Ghareeb, A. E., et al. (2026). [A multi-agent system for automating scientific discovery (Robin)](https://www.nature.com/articles/s41586-026-10652-y). Nature, 655, 497–505.
<a id="reference-39"></a>
39. Gupta, R., Hartford, J., & Liu, B. (2025). LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet? Findings of EMNLP.
<a id="reference-40"></a>
40. Li, S., et al. (2026). [Exploring LLMs for Scientific Information Extraction Using the SciEx Framework](https://arxiv.org/abs/2512.10004). arXiv:2512.10004; AAAI 2026 KGML workshop.
<a id="reference-41"></a>
41. Ríos-García, M., et al. (2026). [AI Scientists Produce Results Without Reasoning Scientifically](https://arxiv.org/abs/2604.18805). arXiv:2604.18805.
<a id="reference-42"></a>
42. Bisht, H., Kumar, V., Jablonka, K. M., Mausam, & Krishnan, N. M. A. (2026). [Agentic AI Scientists Are Not Built for Autonomous Scientific Discovery](https://arxiv.org/abs/2605.08956). arXiv:2605.08956.
<a id="reference-43"></a>
43. Jiang, B. (2026). [HindSight: Evaluating LLM-Generated Research Ideas via Future Impact](https://arxiv.org/abs/2603.15164). arXiv:2603.15164, v2, 17 March 2026.
<a id="reference-44"></a>
44. Hagen, T., et al. (2026). [Investigating Counterclaims in Causality Extraction from Text](https://arxiv.org/abs/2510.08224). arXiv:2510.08224; v2 revised 7 January 2026.
<a id="reference-45"></a>
45. Javaji, S. R., et al. (2025). [Can AI Validate Science? Benchmarking LLMs on Claim→Evidence Reasoning in AI Papers (CLAIM-BENCH)](https://aclanthology.org/2025.ijcnlp-long.127/). IJCNLP-AACL 2025, 2355–2379.
<a id="reference-46"></a>
46. Ge, Z., et al. (2025). [Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs (CONFACT)](https://www.ijcai.org/proceedings/2025/1073). IJCAI 2025.
<a id="reference-47"></a>
47. Yoon, Y., Jung, J., Yoon, S., & Park, K. (2025). [Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion](https://aclanthology.org/2025.findings-acl.980/). Findings of ACL 2025.
<a id="reference-48"></a>
48. Sadhu, J., Shahad, S., & Marino, K. (2026). [DeReLab: Probing Defeasible Reasoning and Confirmation Bias in LLMs with a Generative Benchmark](https://arxiv.org/abs/2608.30413). arXiv:2608.30413; authors report acceptance to EMNLP 2026.
<a id="reference-49"></a>
49. Samanta, A., et al. (2026). [BayesBench: Evaluating LLM Belief Trajectories Under Multi-Turn Evidence Accumulation](https://arxiv.org/abs/2606.30850). arXiv:2606.30850.
<a id="reference-50"></a>
50. Si, C., Yang, D., & Hashimoto, T. (2025). [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](https://arxiv.org/abs/2409.04109). ICLR 2025.
<a id="reference-51"></a>
51. Wang, Q., Downey, D., Ji, H., & Hope, T. (2024). [SciMON: Scientific Inspiration Machines Optimized for Novelty](https://arxiv.org/abs/2305.14259). ACL 2024.
<a id="reference-52"></a>
52. Mukherjee, S., Hamna, H., Bali, K., & Sitaram, S. (2026). [The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment](https://arxiv.org/abs/2606.03043). arXiv:2606.03043; authors report acceptance to EMNLP 2026.
<a id="reference-53"></a>
53. Nguyen, B., et al. (2026). [ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences](https://arxiv.org/abs/2602.11354). arXiv:2602.11354; KDD 2026 AI4Sciences acceptance reported by the authors.
<a id="reference-54"></a>
54. Majumder, B. P., et al. (2025). [DiscoveryBench: Towards Data-Driven Discovery with Large Language Models](https://arxiv.org/abs/2407.01725). ICLR 2025.
<a id="reference-55"></a>
55. Chen, Z., et al. (2025). [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://arxiv.org/abs/2410.05080). ICLR 2025.
<a id="reference-56"></a>
56. Yang, Z., Zhang, C., Zhang, Y., & Wang, H. (2026). [TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents](https://arxiv.org/abs/2609.05079). arXiv:2609.05079, v2, 8 September 2026.
