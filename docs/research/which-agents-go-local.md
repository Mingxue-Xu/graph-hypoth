# Which Agents Go Local? Small Local Models and Retrieval for the Claim Graph Pipeline

*Complete report. A four-page decision version is at [which-agents-go-local-decision.md](which-agents-go-local-decision.md).*

## Abstract

We ask which agents of the claim graph pipeline could run on small local open-weight models (roughly 4–32B parameters, served through vLLM, Ollama or llama.cpp) instead of frontier cloud models, what each swap would do to system output, and whether retrieval can offset what a smaller model loses. Three considerations organise the answer. First, related benchmarks motivate specialised smaller models for extraction, rewriting and some grading tasks, but generic rubric performance does not establish reliability on scientific appraisal, and long-context synthesis remains demanding without a demonstrated 4–32B capability boundary. Second, retrieval, graph construction, generation and criticism jointly affect hypothesis quality; the Synthesist is the principal generator, and panel feedback can lead it to derive new candidates. Third, evaluation must measure scientific accuracy alongside calibration drift, shared judge errors, and schema-valid but incorrect content, because the deterministic core converts a weaker backend's errors into silent distribution shifts rather than visible failures. On retrieval, the literature favours task-dependent retrieval over generic RAG: literal-span tasks are the strongest candidates for lexical retrieval or exact lookup, paraphrase-sensitive tasks generally require hybrid retrieval and reranking, and no study evaluates the complete pipeline end to end. We give provisional per-agent configurations, a capacity ranking with deployment categories, a four-tier roadmap, and the role-level and end-to-end measurements needed before any deployment decision. No numerical range or model pairing here establishes reliability by itself.

## 1. Question, scope and method

**The question.** For each model-backed role in the pipeline, can a small local model replace the current frontier backend at an acceptable error rate, and can dense, lexical, hybrid or structured retrieval be combined with the role's work to make a local deployment more accurate or cheaper? The roles are Extraction, the Evidence Reviewer, the Synthesist, the Critic panel, the Elaborator, the Reader Translator, the Translation Verifier, the Experiment Designer, the Experiment Validator, and two proposed roles, the Profile Compiler and the Result Interpreter.

**Terms.** "Local" describes where inference runs; "open-weight" describes weight availability. Neither establishes capability, licensing freedom or serving cost, and a comparison with a historical cloud checkpoint does not establish parity with the current frontier. The candidate range is 4–32B parameters, with smaller specialist encoders considered separately. **Retrieval** selects records from a corpus; **lookup** addresses an exact or keyed store; **verification** scores whether evidence supports a claim; **orchestration** schedules, delivers and validates those operations. Cost claims refer to the cited study's metric; marginal compute excludes index construction unless stated, and API dollars, tokens, latency, hardware cost and energy are not interchangeable.

**Evidence labels.** **Direct** means the cited study tests the stated mechanism on substantially the same task. **Near analogue** means it tests the mechanism in a different task, domain, model tier or pipeline. **Mechanistic inference** combines separately demonstrated components. **Proposal** means an integration or deployment policy that has no end-to-end evaluation. Almost every result below is a near analogue, and no reviewed study evaluates the complete pipeline or these exact composite roles end to end. A benchmark result can support a candidate without validating its deployment.

**Deployment statuses.** **Candidate** means suitable for evaluation, not approved for immediate substitution. **Conditional candidate** additionally requires the stated controls. **Retain pending evaluation** is a provisional policy, not evidence that local deployment is impossible. All rows require held-out evaluation against an explicit acceptable error rate.

**Method.** Each agent's task contract was reconstructed from public code constants, shipped configuration and documentation. Literature review covered the small-model landscape, agent backends and heterogeneous role assignment, structured extraction, evidence verification, judge scale effects, hypothesis-generation systems, creativity and knowledge scaling, scientific reasoning and experiment design, distillation, judge error correlation, multilingual and reader-facing agents, numeric artifact parsing, and retrieval. Findings were mapped onto agents by the author. The review did not record a reproducible search protocol, screening flow or formal quality appraisal. Several compressed statements in early notes mixed tasks and metrics or overstated transfer; the corrected forms are incorporated below and noted where they change a figure.

**Engineering scope.** The [role configuration](../../src/config.py) already supports selecting compatible backends, including inherited defaults. Specialist encoders, proposed roles, new retrieval components and additional validation controls require engineering. Hardware, context length, concurrency and inference budget must be specified before judging local feasibility.

## 2. Why the architecture shapes the answer

Five properties of the as-built system govern substitution.

1. **Deterministic commit authority.** The [graph commit validator](../../src/validator.py) enforces five checks: schema validity, resolvable references, a current base-state hash, an allowed status transition, and an unused idempotency key. These protect structural and state consistency; they do not establish scientific truth. Schema-constrained decoding can satisfy the first check and none of the others. The commit boundary mitigates some coordination and state-management failures in the MAST taxonomy of multi-agent failures, but reasoning failures, omitted evidence, poor task decomposition, communication failures and inadequate verification remain possible <sup>[1](#reference-1)</sup>. The model-backed Experiment Validator is a separate reviewer, whose current bounded loop does not require a passing assessment of the final plan (§5.7).
2. **Agents grade; rules decide.** The Evidence Reviewer supplies grades to deterministic verdict fusion, which uses a 0.5 floor and a 0.025 near-tie margin. Critic grades feed median aggregation, and Borda fusion combines pool rankings. Deterministic aggregation still depends on accurate scientific judgments and appropriate score calibration; aggregating scores does not remove the need for discrimination, a constant base-rate probability can be calibrated while being useless for selecting individual cases, and discrete rubric anchors mapped to floats do not automatically become calibrated probabilities. Frontier agreement alone cannot validate either property.
3. **Retrieval-grounded prompts.** Evidence anchors are self-contained; the Synthesist may quote only in-context passages; the Designer must ground materials in retrieval or abstain. The prompts were written to minimise dependence on knowledge stored in the model's weights, which is the property that makes small backends conceivable.
4. **Strict JSON everywhere.** This interacts badly with small models (§3): a capacity-dependent accuracy tax, reduced output diversity, and content errors hidden behind schema validity.
5. **Judge complementarity.** Capability and complementary errors are separate properties. Measure both on this task, explicitly configure the intended panel members, and audit the actual escalation conditions (§5.4); different vendors or parameter counts do not guarantee independent mistakes.

**Cost structure and wiring.** In the worked example (20 edges, 8 candidates, 2 judges, 5 confirmed hypotheses), the Evidence Reviewer makes about two-thirds of core calls and the Synthesist holds the deepest context, so those two roles bound the savings. Actual savings require measured input and output tokens, latency and serving costs. Two wiring facts matter for any substitution experiment: the runtime selects `builder`, `critic_panel` and `skeptical_verifier` in that order, so without a dedicated `critic_panel` block the second and third Critic seats inherit the same backend, and changing `builder` also changes the first Critic seat and Extraction unless their wiring is separated. A role-substitution experiment must therefore record every effective backend.

## 3. What small models can and cannot do

**The capability ladder has compressed, unevenly.** Reasoning-distilled and reasoning-tuned models in the 14–32B range report graduate-level science scores above a non-reasoning frontier checkpoint of 2024: DeepSeek-R1-Distill-32B reports 62.1 on GPQA-Diamond against 53.6 for GPT-4o, and Phi-4-reasoning-plus at 14B reports 69.3 <sup>[2](#reference-2), [3](#reference-3)</sup>. Within 4–32B, post-training matters more than parameter count; the Qwen3 report shows thinking mode moving the 8B model from about 30 to about 80 on AIME <sup>[4](#reference-4)</sup>. Instruction following is nearly flat across one family on one benchmark, with Gemma 3's 4B, 12B and 27B instruction-tuned models scoring 90.2, 88.9 and 90.4 on IFEval <sup>[5](#reference-5)</sup>; this does not show that instruction following saturates universally at 4B, and comparisons between tuned and untuned models change training and sometimes inference compute, so they do not isolate parameter count. A model's published score may also use substantial reasoning tokens or repeated sampling; identify the checkpoint and evaluation date rather than treating an older frontier comparison as a generic 2026 frontier comparison.

**Structured output needs task-specific evaluation.** The Format Tax and Capacity, Not Format studies motivate separating reasoning from formatting; effects depend on model, schema, task and token budget, and the latter does not test grammar-constrained decoding <sup>[6](#reference-6), [7](#reference-7)</sup>. In the Constraint Tax's main sub-3B aggregate, a hard answer-only schema moved schema validity from 61.5% to 100%, answer accuracy from 19.7% to 11.0%, executable accuracy from 12.0% to 11.0%, and wrong-but-schema-valid outputs from 49.5% to 88.9%; the increase in valid-but-wrong outputs partly reflects malformed outputs becoming valid, and these are different metrics rather than a single 39-point accuracy drop <sup>[8](#reference-8)</sup>. A study of 44 models reports about 0.22 bits less mean answer-choice surprisal and about 0.20 bits less field entropy under structured output on word-choice prompts with multiple valid answers <sup>[9](#reference-9)</sup>; this motivates testing JSON's effect on Critic grading diversity and escalation, and does not demonstrate either. Compare free-form reasoning followed by serialisation with a single-pass interface on the actual schemas and token budgets, including the extra tokens, latency and possible serialisation errors.

**Quantisation and usable context require measurement.** The reasoning-under-quantisation study finds useful weight-only 4-bit configurations, with observed quality changes of a few percentage points that vary across models, methods and tasks, larger costs for 4-bit activations or KV cache and for 3-bit weights, and greater degradation on hard tasks <sup>[10](#reference-10)</sup>. Weight-only 4-bit is a candidate configuration, not the only permissible one, and "near-free" describes an observed quality change, not total serving cost. RULER defines effective context using a particular benchmark threshold, and its ratios vary by model; NoLiMa uses another criterion and reports Llama 3.1 8B falling from 76.7% at short context to 14.2% at 32K tokens on latent-association retrieval <sup>[11](#reference-11), [12](#reference-12)</sup>. Usable context can be substantially smaller than the advertised maximum; measure accuracy against length on each role's task instead of applying a universal fraction.

**Serving and efficiency need a workload definition.** Continuous batching with cached grammars suits the high-concurrency Evidence and Critic calls; single-call agents can run on lighter servers. Engine choice matters: JSONSchemaBench distinguishes schema coverage from compliance on supported schemas and reports engine-dependent compliance, and the project's own trial found one engine timing out on this pipeline's enum- and array-heavy shapes <sup>[13](#reference-13)</sup>. Benchmark the actual delta schemas against candidate grammar engines before choosing, reuse schema objects for cache hits, and report both coverage and compliance with engine versions. Any efficiency claim must record model and quantisation versions, hardware, weights plus KV cache and runtime memory, input and reasoning tokens, concurrency, utilisation, retries, latency, and cost per accepted result. A low training-run bill excludes substantial costs unless labelling, teacher calls, evaluation and tuning are explicitly included, and call counts alone do not identify the most expensive role.

## 4. Verdict summary

| Agent | Task | Proposed status | Local candidate | Main condition and dominant risk |
|---|---|---|---|---|
| Extraction | Seed → typed nodes and edges (strict JSON) | Candidate | Task-tuned 7–9B or a specialist encoder | Constrained decoding; compare specialist and generative extractors on exact spans, relation direction and domain transfer |
| Evidence Reviewer | Grade passages per edge on entailment and five design-quality axes | Conditional candidate | 14–32B reasoning-tuned; specialist NLI benchmarked separately | Evaluate entailment and study-design assessment separately; measure downstream verdict errors on adjudicated edges |
| Synthesist | Three-turn thread: mine, propose, revise, with verbatim quotes | Retain pending evaluation | Specialised 4–32B alternatives for comparison | Compare on hypothesis quality, evidence coverage and quote fidelity with equivalent evidence access |
| Critic | Two base judges plus optional third: novelty, saturation, mechanism and terminology audits, ranking | Conditional candidate | One 32B reasoning-tuned seat beside a strong reviewer | Measure complementary errors first; audit agreed decisions as well as escalations |
| Elaborator | Hypothesis → plain-language reader card | Candidate | About 7B | Test semantic preservation separately from fluency and readability |
| Reader Translator | Reader-vocabulary rewriting | Candidate | 7–13B translation models on supported language pairs | Evaluate each language pair, technical vocabulary and passage length |
| Translation Verifier | Faithfulness check of the rewrite | Conditional candidate | Quality-estimation models | Measure segment-level omissions and meaning changes; generation scores do not validate a verifier |
| Experiment Designer | Confirmed hypothesis → grounded experiment plan | Conditional candidate | 32B start; 14B lower-cost candidate; 7B exploratory baseline | Structured grounding; initially retain the frontier Experiment Validator; add final-plan review |
| Experiment Validator | Five-criteria plan grading plus citation-grounding check | Conditional candidate | 8–32B judge-tuned | Test false acceptance and rejection against independent judgments; the current loop is feedback, not an acceptance gate |
| Profile Compiler (proposed) | Multilingual narrative → typed profile and claims | Conditional candidate | Fine-tuned extractor | Test multilingual and compositional inputs; user confirmation is an additional check |
| Result Interpreter (proposed) | Metrics tables and logs → typed evidence | Conditional candidate | Code-specialised local model behind numeric gates | Validate source coordinates, outcome labels, units, arms and time points; recompute derived values |

## 5. Agent-by-agent analysis

### 5.1 Extraction

**Task and evidence.** One call per run, a closed ten-type vocabulary, no truth assessment. This is the literature's strongest small-model regime: a 0.3B GLiNER encoder beats ChatGPT by 13 F1 on out-of-domain named-entity recognition, a 440M multi-task extractor beats Llama-3-8B by 43 F1 on relation extraction, guideline conditioning adds about 13 F1 where scaling from 7B to 34B adds under two, and QLoRA-tuned models of 3B or less beat frontier zero-shot on seven relation-extraction benchmarks <sup>[14](#reference-14), [15](#reference-15), [16](#reference-16), [17](#reference-17), [18](#reference-18)</sup>. Three bounds apply. The pattern requires fine-tuning, since zero-shot models of 27B still trail GPT-4-class models by double digits on open-schema extraction. Causal relation extraction carries substantial task- and metric-dependent error: in PubMedCausal, PubMedBERT scores 0.7391 F1 on detecting causal language, while DeepSeek-R1-32B scores 0.6765 cosine-based pair F1, 0.5758 token-overlap pair F1 and 0.2383 exact pair F1 on extracting cause–effect pairs, which are different tasks and metrics rather than a shared ceiling <sup>[19](#reference-19)</sup>; the common shorthand that causal extraction "caps near 0.68–0.74 F1 for everyone" conflates them. And fine-tuned extractors can collapse under domain shift, with one cited extractor falling from 87% to 40% F1 out of domain; this matters for arbitrary seed fields. Errors here are visible early and cheap to redo.

**Retrieval channels to test.** Four single-shot, deterministic-core-driven lookups can be assembled into one inline prompt, which avoids an open-ended search loop and follows the delivery pattern the anchor study favours (§8). First, retrieved demonstrations instead of a static few-shot set: on two relation-extraction evaluations, contrastive demonstration retrieval raised Llama-3.1-8B F1 from 15.7 to 38.7 and from 8.3 to 21.5, relative gains of roughly 146% and 159% from low baselines, while gains at 70B were about half that in relative terms <sup>[20](#reference-20)</sup>. Second, schema-slice retrieval that narrows the closed schema shown to the model: a trained schema retriever raised a 1B model's F1 from 0.23 with a BM25 baseline to 0.42 with latency savings from caching, and a separate frontier-scale study reported gains in F1 with roughly halved latency and tokens; the 4–14B range was not tested <sup>[21](#reference-21), [22](#reference-22)</sup>. For this pipeline's fixed vocabulary, deterministic dictionary lookup can inject the relevant type guidelines. Third, verbatim-span grounding with deterministic verification: a filings-extraction system that required at least 40% of a quote's four-word runs to appear verbatim in the source, retrying otherwise, reduced its source-reported unsupported-claim rate from 8% to approximately 0%, and a schema-grounded clinical extractor with validator-in-the-loop repair raised structural validity from 0.51 to 0.96 in a median of 1.2 repair rounds <sup>[23](#reference-23), [24](#reference-24)</sup>. Neither measures the recall cost of forced verbatim spans, so track true claims lost for lack of clean contiguous support. Fourth, node deduplication against the existing graph as a cascade: exact or alias-table lookup, BM25 blocking to a shortlist of about five to ten, an optional compact bi-encoder such as SapBERT for paraphrase or cross-lingual variants, and small-model adjudication over the shortlist. Fine-tuned 7–13B matchers met or beat GPT-4 on most evaluated entity-matching benchmarks, a 124M model came within 4.4 F1 points of GPT-4 at a source-reported inference cost thousands of times lower under that paper's assumptions, locally served Qwen3-4B and 8B gained 2.6–24.2 entity-matching F1 points from retrieval augmentation, and bounded shortlist selection added 19–32 points at 7–8B in a separate study <sup>[25](#reference-25), [26](#reference-26), [27](#reference-27), [28](#reference-28), [29](#reference-29)</sup>. Entity-matching evidence comes from e-commerce and bibliographic benchmarks, not claim-graph concept vocabularies.

### 5.2 Evidence Reviewer

**Task and evidence.** This volume agent grades each passage's entailment and contradiction maps and five design-quality axes, then lists cross-passage risks and qualifiers; deterministic fusion selects the verdict. The two parts require separate evaluation. Fine-tuned NLI and claim-verification studies motivate smaller entailment models <sup>[30](#reference-30)</sup>, but the full prompt also asks about causal design, measurement validity, population fit, confounder adjustment and bias, and a self-contained rubric does not make those judgments simple. Reported agreements for model-based risk-of-bias assessment range roughly from κ 0.2 to 0.6 across settings, and human–human agreement of about κ 0.40 in another setting is not a ceiling for model–human agreement here: annotation protocols, label prevalence and task mix differ, and poor performance by a larger model does not establish adequate performance by a smaller one. Evaluate 14–32B reasoning-tuned models for the full role; this is a candidate range motivated by related benchmarks, not a validated requirement. Generic rubric benchmarks do not establish suitability, since fine-tuned judges can generalise poorly beyond their training tasks <sup>[31](#reference-31)</sup>. Replay can measure grade and verdict changes; trusted labels are needed to decide which changes are improvements. POPPER reports Type-I error of 0.103 with Sonnet 3.5 versus 0.230 with Haiku 3.5 after changing its backbone, which illustrates that a backbone change can raise observed false positives in a system that designs and executes statistical tests; it is not an isolated Evidence-role calibration ablation <sup>[32](#reference-32)</sup>.

**Retrieval and delivery upgrades to test.** Along the existing enrichment, triage and review stages: add a BM25 channel beside the dense channel, fuse with reciprocal-rank fusion, and rerank with a small cross-encoder before the triage cutoff, since on a scientific-claim retrieval task hybrid plus reranking reached Recall@5 of 0.816 versus 0.587 for dense-only <sup>[33](#reference-33)</sup>; apply moderate context pruning between triage and the prompt, where extractive pruning was accuracy-neutral or positive and compression to 5–10% of tokens cost under 10% relative accuracy in the cited evaluations, validating reader, dataset and compressor together and capping compression where the five axes degrade <sup>[34](#reference-34), [35](#reference-35)</sup>; test a lightweight NLI offload for the entailment map, since an off-the-shelf NLI evaluator reached 83.57% accuracy versus GPT-4o's 83.60% on DIVER-QA and NLI plus a lexical-match feature reached 84.50%, a reference-based QA-evaluation result rather than long-context scientific entailment <sup>[36](#reference-36)</sup>; and activate the specification's currently unused passage-budget and scheduling helpers rather than reviewing every triaged candidate. The model-side constraint is utilisation: for untuned models at 7B or below, Pandey identifies severe failures to use even oracle context, and does not test 14B <sup>[37](#reference-37)</sup>; PRGB reports Qwen2.5 scaling gains that vary by language and subtask, with the 7B to 32B sequence rising from 49.3 to 58.8 on its Chinese aggregate and from 63.16 to 66.70 on its English aggregate <sup>[38](#reference-38)</sup>. Treat 14B as a candidate tier requiring direct in-domain comparison with 7B and 32B, not a literature-established threshold. The anchor study's scope does not support lexical-only retrieval as the default for this paraphrase-tolerant grading; benchmark hybrid retrieval with lexical retrieval as a complementary channel for named concepts.

### 5.3 Synthesist

The persistent three-turn task combines concept mining, cross-concept hypothesis generation, mechanism chains, faithful quotation and revision. These are reasons to retain a capable backend provisionally. NoLiMa's 8B result does not establish failure across the entire 4–32B range or on this scientific task <sup>[12](#reference-12)</sup>; conversely, OpenScholar demonstrates strong literature synthesis with a specialised 8B retrieval system without validating novel hypothesis generation here <sup>[39](#reference-39)</sup>. Retain the frontier Synthesist provisionally while evaluating specialised 4–32B alternatives with equivalent evidence access and recorded inference budgets. Measure hypothesis quality, evidence coverage, quote fidelity and performance across context lengths; test passage re-injection or chunking rather than assuming a persistent conversation preserves recall.

The Synthesist is the principal generator, but achievable quality also depends on available evidence, graph construction, and the Critic's selection and feedback. In the [implemented cycle](../../src/cycles/synthesist_cycle.py), the Synthesist repairs flagged candidates and derives new candidates from the panel's critique before a second panel review, so criticism can cause additional ideas to enter the pool. This does not prove that any role dominates measured outcomes; evaluate the components and their interactions through role substitutions and end-to-end results. ResearchAgent provides a related example of literature-conditioned generation with iterative review, not a causal ranking of this pipeline's roles <sup>[40](#reference-40)</sup>. For novelty assessment specifically, the reviewed systems route cross-paper comparative synthesis through 70–120B-class models with at most a 14B model drafting; that is an absence of a direct small-model result, not evidence of impossibility.

### 5.4 Critic

Each judge, in one listwise call, grades field-relative novelty and saturation, audits each mechanism step and the terminology, ranks the pool and critiques it. The verdict splits.

The **audit half** is a candidate for local evaluation, but mechanism validity and scientific terminology require domain understanding, and generic judge results, such as an 8B judge's reward-benchmark performance or a 32B judge's agreement with GPT-4o, motivate testing without establishing correct scientific audits. Because the evidence is already in context, the candidate mechanism is a two-stage cascade rather than corpus search: a deterministic exact or fuzzy substring check that each quoted span exists in the source, then a small specialised entailment model judging whether the audited step is semantically supported. Vectara reports that its 0.1B HHEM-2.1-Open achieved higher balanced accuracy than zero-shot GPT-3.5 and GPT-4-0613 on three English hallucination-detection benchmarks, with margins over GPT-4 of 0.17–2.64 points, a vendor-reported result requiring in-domain validation <sup>[43](#reference-43)</sup>; separately, a fine-tuned 770M model outperformed zero-shot GPT-3.5 with chain-of-thought on attribution judgment, while out-of-domain attribution accuracy for generically prompted small models fell below 60% <sup>[44](#reference-44)</sup>. These gains depend on task-specific fine-tuning or purpose-built checkpoints.

The **novelty half** requires field knowledge and a suitable literature reference set. Reported novelty benchmarks show uneven model rankings and substantial elicitation effects without establishing a reliable size ordering. A retrieve-then-rerank novelty checker reported about 13% higher agreement with its reference judgments, and a literature-grounded review agent reported an evidence-based-critique score of 1.48 versus 0.10 for GPT-4o <sup>[45](#reference-45), [46](#reference-46)</sup>; both support retrieving and reranking candidate prior work before comparative synthesis, and both used large models for the synthesis itself. Evaluate novelty judgments separately from mechanism and terminology audits, including whether the panel demotes useful new ideas or promotes established relationships.

**Capability and complementary errors are separate properties.** Kim et al. report about 60% agreement conditional on both models being wrong on one leaderboard dataset, where the one-third comparison assumes uniform choice among three incorrect options, and they find shared errors among large, accurate models across providers <sup>[41](#reference-41)</sup>. That is neither an error rate for two local scientific judges nor evidence that scale or vendor diversity ensures independence; CAPA motivates measuring functional similarity directly <sup>[42](#reference-42)</sup>. Shared training and self-preference are risks to measure, and deployment location does not remove them. The proposed starting configuration is to pilot one 32B reasoning-model seat with a strong reviewer, then evaluate a third reviewer where escalation is enabled, selecting panel members by measured task-specific mistakes, auditing some agreed decisions, and comparing alternative pairings including all-local panels before drawing deployment conclusions. The [current escalation function](../../src/cycles/panel.py) consults a third judge only when base judges disagree on the first-ranked candidate or on the `already_established` or `not_judgeable_by_field` flags; numerical-grade, mechanism-audit and terminology-audit disagreements do not independently trigger it. Evaluate whether serious mechanism or terminology disagreements should trigger escalation; any revised rule requires implementation and validation.

### 5.5 Elaborator, Reader Translator and Translation Verifier

All content is supplied and the output is presentation-layer, but presentation failures are not necessarily visible or confined to fluency: a fluent rewrite can change negation, uncertainty, scope, numbers or causal strength without looking wrong, and the Translation Verifier is an evaluator rather than a presentation component. ReLay reports model- and prompting-dependent faithfulness differences, with GPT-4o's factuality score falling from 0.671 without personalisation to 0.515 under metadata prompting, so the risk is task-inherent rather than a small-model artefact <sup>[47](#reference-47)</sup>. Specialised 7–13B translation models sit within roughly 0.5–1 COMET of GPT-4 on their ten or so supported languages, with the gap growing with segment length and unproven beyond that set <sup>[48](#reference-48), [49](#reference-49)</sup>; xCOMET, MetricX-24 and CometKiwi motivate testing specialised evaluation components with language-pair and metric-specific assessment <sup>[50](#reference-50), [51](#reference-51), [52](#reference-52)</sup>. On WMT25, open models of 27–30B doing single-pass quality estimation reached system-level soft pairwise accuracy of 0.82 and 0.83, above reference-based COMET22 at 0.73 and reference-free COMETKiwi22 at 0.60 but below GEMBA-MQM V2 at 0.84 and Gemini-2.5-Pro at 0.87, and remained weaker at segment-level ranking and error-span detection, with the best open filtered span F1 for Czech→German at 10.37 versus 18.17 <sup>[53](#reference-53)</sup>. The quality-estimation literature's structural blind spot, meaning-altering omissions, is task-architectural and is not something retrieval addresses on current evidence.

**Retrieval.** For the Elaborator, two placements are supported on source tasks: conditioning lay rewriting on retrieved term definitions improved readability and factual correctness on the CELLS corpus <sup>[54](#reference-54)</sup>, and routing only elaboration sentences, rather than simplification sentences, through a retrieval-augmented QA check raised ROC-AUC from 0.727 to 0.810 relative to its no-retrieval configuration <sup>[55](#reference-55)</sup>. For the Reader Translator, reader-vocabulary substitution is a finite-set, verbatim-span operation: retrieve glossary entries by exact-match lookup rather than vector search, then use trie-based constrained decoding for the substitution field, which restricts that field to retrieved terms without constraining surrounding free text <sup>[56](#reference-56)</sup>; terminology-aware translation studies report improved consistency from glossary retrieval <sup>[57](#reference-57), [58](#reference-58)</sup>. No reviewed paper evaluates glossary lookup plus constrained decoding end to end. For the Translation Verifier, no direct retrieval result was found; do not build retrieval infrastructure here without a pilot, and if the segment-level gap matters, test a retrieve-and-QA pattern experimentally.

### 5.6 Experiment Designer

SCOPE identifies datasets, baselines and metrics as the difficult parts of experiment design. Adding chain-of-thought plus web search produced no significant gain for five of seven models, lowered one flagship model's total score from 18.22 to 16.77, and raised DeepSeek-V3.2's aggregate redline rate from 7.67% to 14.00%; the full OptED workflow reduced that rate to 2.67% and the stage-isolation-only ablation reached 4.33%. Redline rate combines source hallucinations, metric incompatibilities and constraint violations, so it is not a fabrication rate, and OptED combines several workflow changes, so the study does not isolate retrieval as the dominant lever over model capability <sup>[59](#reference-59)</sup>. AutoSDT reports mixed effects for 7B fine-tuning: ScienceAgentBench success fell from 3.3% to 2.3% while valid execution rose from 19.9% to 27.5% and DiscoveryBench hypothesis matching rose from 4.8% to 6.3% <sup>[60](#reference-60)</sup>; these discovery and execution tasks do not establish a hard 14B threshold for this planning role, and the shorthand "zero gain at 7B" overstates them.

For the Designer, start with a 32B reasoning-tuned model, evaluate 14B as a lower-cost candidate, and include 7B as an exploratory baseline; treat approximately 14B as a provisional deployment starting point, not a demonstrated capability floor. Compare candidates with equivalent evidence access and recorded inference budgets. Initially retain a strong independent reviewer while evaluating a smaller Designer; before treating its review as an acceptance safeguard, review the final plan and define explicit acceptance criteria (§5.7). Plans can waste researcher effort even when their graph references and required fields are valid.

**Retrieval.** Structured, curated retrieval improved recommendation in AgentExpt, which built its knowledge base from 108,825 papers across ten AI venues and whose fine-tuned 0.6B retriever and 7B reranker outperformed graph-aware SymTax by 4.46–7.23% relative on Recall@20 and 7.52–8.27% on HitRate@10 depending on the task <sup>[61](#reference-61)</sup>; in a separate structured-target study, grounding raised execution accuracy from 0% exact match to 71–79% on SQL and API generation <sup>[62](#reference-62)</sup>. Use a short deterministic schedule as the initial default for untrained small models: on a 5,000-question HotpotQA sample with Qwen2.5-7B, fixed hybrid reciprocal-rank fusion outperformed a rule-based strategy selector by 1.8 exact-match points, and two fixed retrieval iterations reached 52.9 exact match versus 53.2 at five; the study does not test a model-driven router, and trained small models can operate search loops, so this supports a bounded schedule as a baseline rather than a universal prohibition <sup>[63](#reference-63)</sup>. Enforce citation IDs mechanically and give abstention a schema branch: in one evaluation, mechanical citation-ID overlap checking reached 92% citation accuracy with zero hallucinated citations, whereas the RAG-only systems fabricated 17% of references <sup>[64](#reference-64)</sup>, and because prompt-based abstention fails under misleading retrieved context <sup>[65](#reference-65)</sup>, test an enforced "no passage found" field together with evidence-conflict checks under both missing and misleading evidence. The proposal is a deterministic pre-retrieval stage over curated structured sources feeding the planner inline, with retrieval isolated in its own schema-constrained step and the Validator checking what enforcement misses; the source results are study-specific reference points, not expected effect sizes.

### 5.7 Experiment Validator

Evaluate 8–32B judge-tuned models for the Experiment Validator; the range is motivated by related benchmarks and does not establish reliability on this pipeline's scientific task. Its five criteria can require substantial reasoning: causal identification, operationalisation, confounder control, robustness, feasibility and reproducibility. Test these capabilities and citation grounding separately, measuring false acceptance and rejection against independently adjudicated cases; agreement with a frontier judge is a diagnostic, not ground truth, and generic judge performance does not establish that a cited passage supports a particular dataset or metric. In the [current implementation](../../src/cycles/experiment.py), the model Validator gives revision feedback: with the default one refinement round it reviews the initial plan and can return the revised plan without reviewing it again, with zero refinement rounds it does not run, and the final return checks content completeness while the commit validator checks structural and state invariants. Neither requires a passing scientific assessment of the final plan, so the loop is feedback, not an acceptance gate.

**Retrieval.** The citation-support check follows a retrieve-then-verify pattern. In FActScore's biography evaluation, retrieval reduced the ChatGPT evaluator's aggregate estimation error from 39.6 to 5.1 points on InstructGPT generations, and an instruction-tuned 7B estimator obtained aggregate errors of 1.4, 0.4 and 9.9 points across three generators against 5.2, 4.7 and 8.7 for ChatGPT; these are aggregate estimation errors, not per-claim verification-error rates, so reading them as verifier error rates would be wrong <sup>[66](#reference-66)</sup>. The mechanism to test is the ALCE and AutoAIS pattern: a fixed-purpose NLI scorer over cited-passage and claimed-dataset pairs, with a deterministic lexical pre-filter for literal dataset names and numbers before any model call <sup>[67](#reference-67), [68](#reference-68)</sup>. Grounding grading criteria in retrieved domain literature raised criteria alignment with expert gold standards by 18–44 accuracy points in one study <sup>[69](#reference-69)</sup>. Two caveats: retrieval utility varied from +0.68 to −0.38 with corpus–claim alignment in a biomedical claim-verification study, so do not add an unvalidated second open-domain search pass <sup>[70](#reference-70)</sup>; and retrieval-augmented small judges remained less reliable than large ones in the cited evaluation of judges for evidence-based research agents <sup>[71](#reference-71)</sup>. The deterministic core should remain the sole component permitted to accept and persist changes.

### 5.8 Profile Compiler (proposed)

Multilingual comprehension carries a scale penalty, with 8–12B models about ten points below 70B-plus models on Global-MMLU and much higher variance on low-resource languages <sup>[72](#reference-72)</sup>, and prompted generative models of any size lose 7.5–40 points to small supervised encoders on multilingual structured extraction, while task-specific fine-tuning nearly closes the gap in the one direct study <sup>[73](#reference-73)</sup>. Viability is therefore an architecture decision rather than a size decision: only as a fine-tuned extractor. The researcher confirms the compiled profile before the run, which caps the damage without removing it. Three retrieval sub-answers apply. Span grounding transfers from Extraction, with the cited systems reporting about 90% verbatim-match rates and schema validity rising from 0.51 to 0.96 with validator-in-the-loop repair, and copy-constrained decoding plus a 365-example preference-tuning pass raising Llama-3-8B contextual faithfulness to 92.8% against GPT-4o's 47.5% in the same regime <sup>[24](#reference-24), [74](#reference-74)</sup>. Lexical retrieval is more robust than the tested dense baselines on code-switched text, where generic multilingual dense embedders degraded more than BM25 (−11.9 versus −6.6 nDCG), one reranker fell by 34 nDCG points, and zero-shot dense retrievers lost to BM25 on low-resource morphologically rich languages <sup>[75](#reference-75), [76](#reference-76)</sup>; use exact or substring lookup for within-document narrative search and benchmark a hybrid model if cross-document retrieval is needed. Terminology normalisation should favour retrieval over parametric recall, since a dense cross-lingual retriever outperformed the best evaluated generative model at zero-shot biomedical concept normalisation across ten languages <sup>[77](#reference-77)</sup>. The composition risk of chaining normalisation, extraction and grounding on one small multilingual model is unmeasured anywhere, so the in-house benchmark recommendation stands.

### 5.9 Result Interpreter (proposed)

This role is deceptively risky. On the closest task match, extracting numeric RCT results into structured form, GPT-4 reached 48.7–65.5% exact match against 8.7–16.4% for 7–13B open models, and the dominant error, a wrong number in the correct format, doubled from 21.9% for GPT-4 to 43.8% for Mistral-7B <sup>[78](#reference-78)</sup>. Code-execution scaffolds only help code-specialised backbones; a general 7B model collapsed to 0.6% in a program-of-thought evaluation. What works is deterministic scaffolding that shrinks the model's decision surface: LibreLog reports that clustering and cache-matching scaffolding improved an 8B model's parsing accuracy by 13.7% relative to prior GPT-3.5- and GPT-4-backed log parsers while running 2.7 times as fast, and LogBatcher's cache-first design roughly halved model invocations <sup>[79](#reference-79), [80](#reference-80)</sup>. With the same 8B backbone, schema-based structured memory exceeded plain RAG by 11.8 accuracy points on single-turn deterministic fact extraction, plain RAG did better on multi-turn aggregation, and task-aware routing added 2.9 points over either fixed choice <sup>[81](#reference-81)</sup>. In a structured-extraction benchmark, wrong values accounted for about 97% of remaining errors after structural errors were removed, and one 12B model outperformed several 70B models <sup>[82](#reference-82)</sup>. Log parsing and financial QA are analogues; none validates scientific result interpretation. Because this agent injects evidence with `own_result` trust directly into the verdict path, local deployment requires validator-side gates: attach each extracted number to source coordinates (document, table, row, column, labels, unit), validate the value and labels jointly, recompute derived quantities, and treat plain string presence as a first-pass check only, since a value can occur in the wrong arm, outcome, time point, subgroup or unit and derived values may correctly differ from the source string. Once gated, the privacy argument, that unpublished results never leave the machine, favours local.

## 6. Model capacity by role and deployment categories

Prioritise strong models for the Synthesist and Critic panel when the goal is creative, grounded hypotheses. The ranking below estimates the capability each of the seven current role contracts needs; it is an architectural judgment, not a measured leaderboard, and the sizes are estimates subject to task-specific validation.

![Seven agent roles plotted by creativity and grounding, with bubble size and colour identifying personal computer, GPU workstation, or hosted frontier service.](figures/agent-model-capacity-map.svg)

*Positions indicate qualitative contributions to generated hypotheses. Bubble size and colour identify deployment categories, not proportional parameter counts or hardware requirements. Critic and Validator placements cover their local candidates; retained strong reviewers may use hosted services.*

| Rank | Role | Main capability required | Proposed model allocation |
|---|---|---|---|
| 1 | Research Synthesist | Integrate papers, generate distinct mechanisms, preserve source meaning, and revise across turns | Retain frontier provisionally; compare specialised 4–32B alternatives |
| 2 | Critic panel | Judge field novelty, audit mechanisms and terminology, and critique the candidate pool | Pilot one 32B reasoning-tuned seat with a strong reviewer; measure complementary mistakes |
| 3 | Experiment Designer | Turn a mechanism into a feasible, falsifiable experiment with appropriate controls and measurements | Start at 32B; evaluate 14B as a lower-cost candidate and 7B as an exploratory baseline |
| 4 | Experiment Validator | Detect flaws in causal identification, feasibility, reproducibility and citation support | Evaluate 8–32B judge-tuned models; initially retain a strong reviewer while testing a smaller Designer |
| 5 | Evidence Reviewer | Assess entailment, study design, confounding, population fit and evidence qualifications | Evaluate 14–32B reasoning-tuned models for the full role; benchmark specialist entailment separately |
| 6 | Extraction | Preserve the seed's concepts, relationship direction, scope and assumptions | Pilot a task-tuned 7–9B model; specialist extractors need integration and whole-contract evaluation |
| 7 | Narrative Elaborator | Explain supplied content while preserving uncertainty and causal meaning | Pilot approximately 7B; evaluate semantic preservation separately from readability |

Critic versus Designer, and Validator versus Evidence, are close comparisons that domain and task scope can reverse; higher reasoning demand does not necessarily imply a larger specialist model or greater downstream influence. On creativity, the Synthesist contributes most directly, followed by Critic feedback and selection, while Extraction supplies the seed structure, the experiment agents contribute experimental ideas after hypothesis selection, and the Elaborator communicates existing content. On grounding, the Synthesist prevents misrepresentation while writing and the Critic independently challenges mechanisms and source interpretations; the [current workflow](../../src/run_path.py) applies Evidence review to seed edges before generation, not to newly generated hypotheses, so adding review of their factual premises would make the Evidence Reviewer a leading grounding component. A novel inference can remain untested without being hallucinated: require accurate source-derived premises, explicit assumptions and a clearly labelled conjecture, and treat crucial facts recalled from model knowledge as needing verification or an uncertainty label, since agreement between models is not proof.

**Deployment categories** are planning targets for the proposed models, not proven hardware minima. Local estimates assume dense models with 4-bit weights, one model resident at a time and low concurrency; context length and runtime overhead add memory.

| Category | Hardware planning estimate | Roles |
|---|---|---|
| Personal computer | CPU with 16–32 GB RAM; GPU acceleration optional for the 7–9B candidates | Extraction, Elaborator |
| GPU workstation | One GPU with 24–48 GB VRAM, or roughly 32–64 GB unified memory with GPU acceleration | Evidence Reviewer, the Critic's local 32B seat, Experiment Designer, Experiment Validator |
| Hosted frontier service | Provider-managed data centre accessed through an API; no local GPU; exact serving hardware unspecified | Synthesist, provisionally |

These are engineering estimates informed by llama.cpp's CPU and GPU support and Qwen's measured memory examples, in which a 32B AWQ model used about 18.7 GB at the shortest tested input and 31.6 GB at roughly 30k input tokens <sup>[83](#reference-83), [84](#reference-84)</sup>. CPU execution of larger models remains possible when memory permits; placement reflects a practical acceleration target, and smaller validated candidates could move roles to the personal-computer category.

## 7. Three silent failure modes

The deterministic core means a weaker backend rarely breaks the system openly; it shifts distributions that fixed rules then act on. Three framing results from the review: role criticality is empirical rather than intuitive, since mis-assigning the strong model swung accuracy by +44 and −36 percentage points in the one rigorous role-assignment study, while the widely quoted estimate that 40–70% of agent calls could move to small models is an expert prior <sup>[85](#reference-85), [86](#reference-86)</sup>; stronger generators are not automatically easier to check, because their errors can be less verifiable, so a frontier builder with a small critic is not automatically safe <sup>[87](#reference-87)</sup>; and weak judges close much of the gap when interaction is structured as debate or consultancy rather than free-form grading <sup>[88](#reference-88)</sup>.

| Failure mode | What can go wrong | Proposed measurement or control |
|---|---|---|
| Score drift | A backend changes score distributions, and therefore fixed-threshold verdicts, even if rankings remain similar; incorrect scientific judgments are a separate risk; Experiment Validator grades currently guide refinement rather than acceptance | Compare discrimination, calibration, false acceptance and false rejection on held-out, independently adjudicated cases; replay measures changes on logged cases but cannot say which are improvements or guarantee coverage of future shifts |
| Shared judge errors | Judges agree incorrectly, so disagreement-based escalation misses the failure; median and Borda aggregation cannot remove a shared bias; the present escalation rule omits substantive audit disagreements | Measure complementary errors and error overlap, audit a sample of agreed decisions as well as disagreements, and evaluate escalation changes; an all-local or cross-vendor label is not a reliability result |
| Valid structure, incorrect meaning | A well-formed claim, quote, number or interpretation is misattributed or misinterpreted; constrained decoding addresses format, not the other four commit checks, and those checks do not establish scientific truth | Check exact source spans and structured provenance; for raw numbers, validate outcome, unit, population, arm and time point; for derived values, retain the transformation and recompute independently; sample schema-valid deltas for content review |

## 8. Retrieval for local agents

This section summarises the retrieval analysis; the full treatment, including what retrieval can and cannot substitute for and why every role still needs a model, is in [Retrieval for Local Agents](retrieval-for-local-agents.md).

### 8.1 The anchor study's framework

Sen et al. compare lexical and vector retrieval inside agentic loops across four harnesses and five models on a 116-question LongMemEval subset, using proprietary models <sup>[89](#reference-89)</sup>. Four results matter here. Inline lexical retrieval beat inline vector retrieval for every tested harness–model pair, for example 93.1% versus 83.6% for Opus 4.6 on the custom harness; the largest margin was Gemini 3.1 Flash-Lite under Chronos at 86.2% versus 62.9%, a 23.3-point gap, while Claude Haiku 4.5 under Claude Code showed 55.2% versus 44.0%, an 11.2-point gap, and the authors' hypothesis that weaker agents are less consistent at iterative query refinement is not established as a monotonic capability effect. Delivery mode can invert the ranking: file-based delivery, where the agent must run extra tool steps to read results, collapsed one strong pairing from 93.1% to 55.2%, because gains from programmatic routing are realised only when the agent reliably closes the loop. The result is scoped to verbatim-span tasks; the authors expect dense or hybrid retrieval to matter more where evidence is rarely literal, as in scientific synthesis over paraphrased abstracts. And retrieval in agents is retrieval plus orchestration: harness and delivery choices moved accuracy as much as the retriever did.

For this pipeline the framework suggests a candidate routing rule. Tasks whose evidence is a literal span (quote existence, alias lookup, glossary substitution, dataset and metric names, numeric cells) are candidates for lexical or exact-match machinery; tasks whose evidence is paraphrased (entailment grading, methods grounding) are candidates for hybrid retrieval with reranking, with context utilisation tested separately; inline delivery is a strong default for untrained small models, and multi-step file-based tool loops should be used only after direct validation or task-specific training. Several heterogeneous studies provide convergent but non-replicative support for lexical or entity-anchored retrieval on literal evidence: CLEAR obtained 0.90 average F1 versus 0.86 for embedding RAG in clinical extraction with 72% lower latency and 71% fewer input tokens <sup>[94](#reference-94)</sup>, and one enterprise-QA benchmark found BM25's advantage over dense retrieval widening as its corpus grew <sup>[95](#reference-95)</sup>. Corpus size alone should not be treated as a universal reason to upweight lexical retrieval, and literal fidelity is not guaranteed by parameter count.

### 8.2 Cross-cutting results

Several findings apply to more than one agent. Long-tail question answering can make retrieval load-bearing: across the cited settings, closed-book accuracy for 7B-class models rose from 5–12% to 44–56% with one gold passage, retrieval-augmented small models outperformed much larger parametric-only models on tail entities, and one study extrapolated that closing its observed gap through scale alone would require roughly 10^15 parameters <sup>[100](#reference-100), [101](#reference-101)</sup>. Passage curation can outweigh retriever swaps: a topically related but incorrect distractor reduced accuracy by as much as 25 points in one setting, and hard-negative retriever training did not eliminate the loss, so reranking or filtering before the model call is a candidate intervention and the contested claim that irrelevant filler helps should not be assumed <sup>[102](#reference-102)</sup>. At the smallest tested scales, context utilisation binds before retrieval quality, as §5.2 notes. Compression gain generally decreases as the uncompressed reader's baseline improves, with strong negative correlations for several compressors and datasets, but no distinct capability breakpoint, so compression should be validated per reader, dataset and method rather than switched by a fixed threshold <sup>[90](#reference-90)</sup>. Misleading retrieved context can impair abstention: across three frozen 3.8B–8B models, explicit abstention prompts still elicited answers on 41.6% of misleading cases although the models usually abstained when evidence was simply missing <sup>[65](#reference-65)</sup>, while AbstentionBench, which is not a retrieval experiment, found little average improvement from scaling Llama 3.1 from 8B to 405B and declines for two reasoning-tuned comparisons <sup>[91](#reference-91)</sup>. Passage placement matters: Lost in the Middle reports 15–30 accuracy-point differences by position, so test best-first and best-last ordering rather than placing the most relevant evidence in the middle <sup>[103](#reference-103)</sup>. And decompose, retrieve and verify per atom is an established baseline pattern in FActScore, SAFE and FacTool, which resembles the pipeline's Evidence and Validator design without validating that composite design end to end <sup>[66](#reference-66), [92](#reference-92), [93](#reference-93)</sup>.

### 8.3 Per-role retrieval components

| Role or operation | Proposed component | Evidence and limitation |
|---|---|---|
| Extraction | Retrieved demonstrations, schema slicing, alias lookup, verbatim-span verification | Relation-extraction and filings-extraction analogues (§5.1); absolute quality and domain transfer require evaluation |
| Evidence Reviewer | Hybrid lexical and dense retrieval, reranking, passage budgeting, specialist NLI verification | Scientific-claim retrieval and QA-evaluation analogues (§5.2); not long-context scientific entailment |
| Critic | Exact support lookup followed by entailment assessment; retrieve and rerank prior work before novelty synthesis | Quote existence and semantic support are separate checks; no direct small-model result for cross-paper comparison |
| Experiment Designer | Entity-linked curated index, small retriever and reranker, short scheduled retrieval, enforced citation IDs | Direct for recommendation, near analogue for planning (§5.6); unrestricted search not supported |
| Experiment Validator | Retrieve supporting passages, then score claim–passage support; lexical pre-filter for literal names and numbers | Aggregate estimation errors, not per-claim rates (§5.7) |
| Elaborator | Retrieve definitions during generation; verify only newly elaborated sentences | Lay-language analogues (§5.5); transfer untested |
| Reader Translator, Profile Compiler | Glossary, alias and schema lookup with constrained decoding; lexical or hybrid retrieval for code-switched text | Mechanistic inference; composite untested |
| Result Interpreter | Template caches, structured lookup, task routing, source-coordinate and recomputation checks | Log-parsing and extraction analogues (§5.9); no validation of scientific interpretation |
| Translation Verifier | No established retrieval addition | Absence of evidence, not evidence of no benefit |

### 8.4 Design rules to test

1. **Route by evidence type.** Verbatim-span work goes to inline lexical or exact-match lookup; paraphrase work goes to hybrid BM25-plus-dense retrieval with a small reranker; finite-set work (glossaries, schemas, aliases) goes to deterministic dictionary lookup that injects the entries relevant to the current item.
2. **Deliver retrieval inline to untrained small models by default.** Use multi-step file-based or model-driven loops only after direct validation or task-specific training; trained small models can operate search loops.
3. **Test offloading sub-tasks to small specialists.** NLI classifiers, rerankers, span locators and entity linkers can match a frontier model on particular tasks; this supports a cascade that removes narrow judgments from the main model, not a claim that every small classifier beats larger generative models.
4. **Shortlist, then adjudicate.** For matching and deduplication, retrieve a candidate set of about five to ten before model adjudication, and validate the candidate-set size and effect in this pipeline.
5. **Verify spans and numbers deterministically, and make abstention a schema branch.** String-match quoted spans; for numbers, validate source coordinates, labels, units and conditions jointly and recompute derived values; require an explicit "nothing retrieved" field plus evidence-conflict checks, and validate missing and misleading conditions separately.
6. **Curate and cap passages, validate compression per reader and dataset, and put evidence for the current claim first.** One near-miss distractor can cost more than a retriever swap gains, and a smaller passage budget can beat a larger one in some settings.
7. **Benchmark 7B, 14B and 32B directly for retrieval-heavy grading.** Severe oracle-context utilisation failures are documented at 7B and below, and no reviewed study establishes a transition at 14B; scaling effects vary by language and subtask.
8. **Give the deterministic core final acceptance authority.** Across the reviewed systems, validators, caches, string checks and fixed schedules often supplied the important reliability controls, and adding unrestricted search to model judgment did not improve most tested planners. This supports testing the pipeline's structured architecture, not treating it as universally validated.

## 9. Roadmap

### 9.1 Tiers

- **Tier 1, pilot first:** Elaborator; Reader Translator on high-resource languages; Extraction with a fine-tuned 7–9B model or a GLiNER-class encoder, with constrained decoding mandatory below about 8B; retrieval embeddings. Candidate families include Qwen3-8B and 14B, Gemma 3 12B and Mistral Small 24B, subject to the licensing checks in §9.3.
- **Tier 2, evaluate scientific review:** Evidence Reviewer (14–32B reasoning-tuned; weight-only 4-bit and vLLM as serving candidates) and Experiment Validator (8–32B judge-tuned), testing scientific reasoning and source support separately against independently adjudicated cases; final-plan review and explicit acceptance criteria before relying on a Designer–Validator pairing; the Profile Compiler as a fine-tuned extractor and the Translation Verifier under their task-specific controls.
- **Tier 3, evaluate conditional configurations:** Critic (one 32B local seat with a strong reviewer; measure complementary errors, agreed mistakes and escalation coverage before choosing the panel); Result Interpreter (code-specialised local model behind numeric gates); Experiment Designer (32B start, 14B lower-cost candidate, 7B exploratory baseline, structured grounding, and an initially retained strong reviewer).
- **Tier 4, retain frontier provisionally:** Synthesist, compared against specialised 4–32B alternatives under equivalent evidence access and recorded inference budgets, including generation–critique interactions in the harness below.

### 9.2 Engineering preconditions

1. Decouple reasoning from serialisation, free-form first and schema second, on every agent that mixes analysis with typed output.
2. Grade in discrete anchors and map them to floats deterministically, without treating the result as a calibrated probability.
3. Evaluate the Evidence fusion thresholds (0.5 floor, 0.025 margin) and score mappings on labelled calibration data, assess any adjustment on separate held-out cases, and measure scientific accuracy as well as calibration.
4. Treat weight-only 4-bit quantisation as the starting configuration and measure alternatives; avoid 4-bit activations or KV cache on sceptical agents without measurement.
5. Benchmark the actual delta schemas against candidate grammar engines before choosing an engine; reuse schema objects for cache hits.
6. Add validator content gates: character-level verbatim-span checks for quotes and numbers, and recomputation of derivable quantities.
7. Measure panel error overlap, audit some agreements, and test whether serious mechanism or terminology disagreements should trigger escalation; evaluate dependence-aware aggregation and confidence-based escalation as candidate policies.
8. Add review of the final experiment plan and an explicit scientific acceptance policy, recording review outcomes separately from content completeness and successful graph commit, and test that the last revision is covered.

### 9.3 Distillation and its legal boundary

Fine-tuning economics are favourable in the closest analogues, where a few hundred to about a thousand examples sufficed and parameter-efficient runs cost from under a dollar to a few hundred dollars <sup>[96](#reference-96), [97](#reference-97)</sup>, and the pipeline's prompt-hashed log is the natural data substrate. The known risks, shortcut learning, out-of-domain collapse and judge brittleness off-distribution <sup>[31](#reference-31)</sup>, are addressable with held-out cross-domain audits. The hard boundary is contractual. Anthropic's output-training policy permits certain non-competing specialised applications, including classification, information extraction and semantic search, while restricting competing-model development <sup>[98](#reference-98)</sup>; the Llama 3.1 Community License permits output-based training subject to conditions, including naming requirements for certain distributed models, and each later release's license must be checked separately <sup>[99](#reference-99)</sup>. Check the teacher's applicable terms and the intended student use, and obtain written permission where the terms are unclear; this is a question for counsel. Policy links were checked on 12 September 2026.

### 9.4 Validation harness

All of the following are buildable from existing assets.

1. **Evidence accuracy and calibration replay.** Re-grade logged edges with 14–32B candidates and the current backend; evaluate entailment and study-design assessment separately; measure verdict flips, threshold sensitivity, false acceptance and false rejection against independent adjudication; keep calibration and evaluation cases separate.
2. **Verbatim-fidelity curve.** Character-diff quoted spans against in-context sources across backends and context lengths, the benchmark the literature lacks.
3. **Critic error-overlap audit.** Compare candidate panels on a labelled pool, including wrong agreements and the serious audit disagreements the current trigger misses; record effective model identities, candidate order and escalation telemetry.
4. **Role-swap and interaction ablations.** Compare the current backend with the proposed per-role candidates, holding evidence access comparable and recording inference budgets; follow isolated substitutions with selected joint changes, especially Synthesist–Critic and Designer–Validator pairings, and measure final hypothesis and plan quality rather than assuming the generator is the sole bottleneck.
5. **Schema-content audit.** Sample schema-valid deltas for content review; track wrong-but-valid rates per agent and backend.
6. **Final-plan acceptance evaluation.** Measure remaining scientific errors after the final refinement, whether the final artifact was reviewed, and false acceptance and rejection under the proposed acceptance policy, including zero-round and exhausted-refinement cases.

The first measurements should be the Evidence replay on independently adjudicated edges and the Critic error-overlap audit. Subsequent deployment decisions should follow their results, with ongoing sampling to detect failures outside the replay set.

## 10. Limitations

No published study tests these exact composite tasks; all numbers are nearest-analogue transfers, flagged per agent, and about a third of the sources are 2025–2026 preprints, with some figures vendor-reported. The report's conclusions rest on convergence across independent sources rather than on single citations, and corrected readings are used where compressed summaries would mix tasks, metrics or transfer claims. The review is targeted and narrative rather than systematic, with no recorded screening flow or formal quality appraisal. Benchmarks measure isolated calls, while system outcome depends on error propagation through rules, loops and human gates, which is what the harness in §9.4 addresses. Domain transfer is a recurring caveat: entity-matching evidence comes from e-commerce and bibliographic benchmarks, span-grounding evidence from clinical and financial extraction, and none from claim-graph concept vocabularies. Model-specific recommendations have short half-lives; the regime-level findings on format, calibration, correlation and long context have been stable across model generations. Finally, this report evaluates whether frontier models are necessary while using frontier-model assistance in the review process; structured extraction contracts, a completeness critique and explicit absence-of-evidence reporting mitigate but do not remove that reflexivity.

## 11. Conclusion

The architecture supports compatible backend substitutions and logged comparisons, but scientific quality still depends on accurate judgments, source coverage and the interactions between roles. Evaluate 14–32B Evidence Reviewers and 8–32B Experiment Validators on independently adjudicated scientific cases. Start Experiment Designer evaluation at 32B, compare 14B as a lower-cost candidate and 7B as an exploratory baseline, and treat approximately 14B as a provisional deployment starting point rather than a demonstrated capability floor, retaining a strong reviewer while implementing and evaluating final-plan review and explicit acceptance criteria. For the Critic, pilot a 32B local seat with a strong reviewer, then select panel members using measured capability and complementary mistakes, including audits of agreements. Retain the frontier Synthesist provisionally while comparing specialised 4–32B alternatives. For retrieval, route by evidence type, deliver inline, verify spans and numbers deterministically, and benchmark 7B, 14B and 32B readers directly. Begin with the Evidence replay against independent judgments and the Critic error-overlap audit, then test role interactions and final-plan acceptance. These measurements can support deployment decisions; no numerical range or model pairing in this report establishes reliability by itself.

## References
<a id="reference-1"></a>
1. Cemri, M., et al. (2025). [Why Do Multi-Agent LLM Systems Fail? (MAST)](https://arxiv.org/abs/2503.13657). NeurIPS 2025; arXiv:2503.13657.
<a id="reference-2"></a>
2. DeepSeek-AI (2025). [DeepSeek-R1](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
<a id="reference-3"></a>
3. Microsoft (2025). [Phi-4-reasoning](https://arxiv.org/abs/2504.21318). arXiv:2504.21318.
<a id="reference-4"></a>
4. Qwen Team (2025). [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). arXiv:2505.09388.
<a id="reference-5"></a>
5. Google DeepMind (2025). [Gemma 3 technical report](https://arxiv.org/html/2503.19786v1), Table 18. arXiv:2503.19786.
<a id="reference-6"></a>
6. Lee, I. Y., D'Antoni, L., & Berg-Kirkpatrick, T. (2026). [The Format Tax](https://arxiv.org/abs/2604.03616v1). arXiv:2604.03616.
<a id="reference-7"></a>
7. Fan (2026). [Capacity, Not Format](https://arxiv.org/html/2606.09410v1). arXiv:2606.09410.
<a id="reference-8"></a>
8. Ray (2026). [The Constraint Tax](https://arxiv.org/html/2605.26128v1), Table 3. arXiv:2605.26128.
<a id="reference-9"></a>
9. [Structured Output Collapses Answer Diversity Across 44 Language Models](https://arxiv.org/html/2607.18476v1). arXiv:2607.18476 (2026).
<a id="reference-10"></a>
10. [Quantization Hurts Reasoning?](https://arxiv.org/html/2504.04823v1). arXiv:2504.04823 (2025).
<a id="reference-11"></a>
11. Hsieh, C.-P., et al. (2024). [RULER](https://arxiv.org/html/2404.06654v1). arXiv:2404.06654.
<a id="reference-12"></a>
12. Modarressi, A., et al. (2025). [NoLiMa](https://arxiv.org/html/2502.05167v1), Table 3. ICML 2025; arXiv:2502.05167.
<a id="reference-13"></a>
13. Geng, S., et al. (2025). [JSONSchemaBench](https://arxiv.org/html/2501.10868v1). arXiv:2501.10868.
<a id="reference-14"></a>
14. Zaratiana, U., et al. (2024). [GLiNER](https://arxiv.org/abs/2311.08526). NAACL 2024; arXiv:2311.08526.
<a id="reference-15"></a>
15. Stepanov, I., et al. (2024). [GLiNER multi-task](https://arxiv.org/abs/2406.12925). arXiv:2406.12925.
<a id="reference-16"></a>
16. Sainz, O., et al. (2024). [GoLLIE](https://arxiv.org/abs/2310.03668). ICLR 2024; arXiv:2310.03668.
<a id="reference-17"></a>
17. Zhou, W., et al. (2023). [UniversalNER](https://arxiv.org/abs/2308.03279). arXiv:2308.03279.
<a id="reference-18"></a>
18. [Sub-Billion, Super-Frontier](https://arxiv.org/abs/2606.22606). arXiv:2606.22606 (2026).
<a id="reference-19"></a>
19. Kunle-John, et al. (2026). [PubMedCausal](https://arxiv.org/html/2605.28363v1), Tables 4–5. arXiv:2605.28363.
<a id="reference-20"></a>
20. [LC-ICL: Label-Guided Contrastive In-Context Learning](https://arxiv.org/abs/2606.29407). arXiv:2606.29407 (2026).
<a id="reference-21"></a>
21. [DLISC: Schema-aware Information Extraction with On-Device LLMs](https://arxiv.org/abs/2505.14992). arXiv:2505.14992 (2025).
<a id="reference-22"></a>
22. [SchemaRAG](https://arxiv.org/abs/2607.00008). arXiv:2607.00008 (2026).
<a id="reference-23"></a>
23. [Grounded Event Extraction from SEC 8-K Filings](https://arxiv.org/abs/2607.08346). arXiv:2607.08346 (2026).
<a id="reference-24"></a>
24. [Schema-Grounded LLM Extraction for FHIR Digital Twins](https://arxiv.org/abs/2601.05847). arXiv:2601.05847 (2026).
<a id="reference-25"></a>
25. Peeters, R., Steiner, A., & Bizer, C. (2024). [Entity Matching using Large Language Models](https://arxiv.org/abs/2310.11244). arXiv:2310.11244.
<a id="reference-26"></a>
26. [AnyMatch: Zero-Shot Entity Matching with a Small Language Model](https://arxiv.org/abs/2409.04073). arXiv:2409.04073 (2024).
<a id="reference-27"></a>
27. [Cost-Efficient RAG for Entity Matching (CE-RAG4EM)](https://arxiv.org/abs/2602.05708). arXiv:2602.05708 (2026).
<a id="reference-28"></a>
28. Wang, T., et al. (2025). [Match, Compare, or Select? (ComEM)](https://arxiv.org/abs/2405.16884). COLING 2025; arXiv:2405.16884.
<a id="reference-29"></a>
29. Liu, F., et al. (2021). [SapBERT](https://arxiv.org/abs/2010.11784). NAACL 2021; arXiv:2010.11784.
<a id="reference-30"></a>
30. Košprdić, M., et al. (2024). [Scientific Claim Verification with Fine-Tuned NLI Models](https://www.scitepress.org/Papers/2024/129000/129000.pdf). SciTePress.
<a id="reference-31"></a>
31. Huang, H., et al. (2024). [Fine-tuned Judge Model Is Not a General Substitute for GPT-4](https://arxiv.org/abs/2403.02839). arXiv:2403.02839.
<a id="reference-32"></a>
32. Huang, K., et al. (2025). [POPPER](https://arxiv.org/pdf/2502.09858v1), Table 4. ICML 2025; arXiv:2502.09858.
<a id="reference-33"></a>
33. [Deep Retrieval at CheckThat! 2025](https://arxiv.org/abs/2505.23250). arXiv:2505.23250 (2025).
<a id="reference-34"></a>
34. [Provence: Efficient and Robust Context Pruning](https://arxiv.org/abs/2501.16214). arXiv:2501.16214 (2025).
<a id="reference-35"></a>
35. Xu, F., Shi, W., & Choi, E. (2023). [RECOMP](https://arxiv.org/abs/2310.04408). arXiv:2310.04408.
<a id="reference-36"></a>
36. Balamurali & Cheng (2025). [Revisiting NLI: Cost-Effective Metrics for QA Evaluation](https://arxiv.org/html/2511.07659v1). arXiv:2511.07659.
<a id="reference-37"></a>
37. Pandey (2026). [Can Small Language Models Use What They Retrieve?](https://arxiv.org/html/2603.11513v1). arXiv:2603.11513.
<a id="reference-38"></a>
38. Tan, et al. (2025). [PRGB Benchmark](https://arxiv.org/abs/2507.22927). arXiv:2507.22927.
<a id="reference-39"></a>
39. Asai, A., et al. (2024). [OpenScholar](https://arxiv.org/html/2411.14199v1). arXiv:2411.14199.
<a id="reference-40"></a>
40. Baek, J., et al. (2025). [ResearchAgent](https://arxiv.org/html/2404.07738v2). NAACL 2025; arXiv:2404.07738.
<a id="reference-41"></a>
41. Kim, Garg, Peng, & Garg (2025). [Correlated Errors in Large Language Models](https://arxiv.org/html/2506.07962v1). arXiv:2506.07962.
<a id="reference-42"></a>
42. Goel, S., et al. (2025). [Great Models Think Alike (CAPA)](https://arxiv.org/abs/2502.04313). arXiv:2502.04313.
<a id="reference-43"></a>
43. Vectara (2024–2025). [HHEM-2.1-Open](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model). Vendor blog.
<a id="reference-44"></a>
44. Yue, X., et al. (2024). [AttributionBench](https://arxiv.org/abs/2402.15089). arXiv:2402.15089.
<a id="reference-45"></a>
45. [Literature-Grounded Novelty Assessment (Idea Novelty Checker)](https://arxiv.org/abs/2506.22026). SDP 2025; arXiv:2506.22026.
<a id="reference-46"></a>
46. [ReviewGrounder](https://arxiv.org/abs/2604.14261). arXiv:2604.14261 (2026).
<a id="reference-47"></a>
47. [ReLay](https://arxiv.org/html/2605.00468v1), Table 2. arXiv:2605.00468 (2026).
<a id="reference-48"></a>
48. Alves, D. M., et al. (2024). [Tower](https://arxiv.org/abs/2402.17733). arXiv:2402.17733.
<a id="reference-49"></a>
49. Xu, H., et al. (2023). [ALMA](https://arxiv.org/abs/2309.11674). arXiv:2309.11674.
<a id="reference-50"></a>
50. Guerreiro, N. M., et al. (2023). [xCOMET](https://arxiv.org/abs/2310.10482). arXiv:2310.10482.
<a id="reference-51"></a>
51. Juraska, J., et al. (2024). [MetricX-24](https://arxiv.org/abs/2410.03983). arXiv:2410.03983.
<a id="reference-52"></a>
52. Rei, R., et al. (2022). [CometKiwi](https://arxiv.org/abs/2209.06243). arXiv:2209.06243.
<a id="reference-53"></a>
53. [CompactQE: Interpretable Translation Quality Estimation via Small Open-Weight LLMs](https://arxiv.org/abs/2605.15763). arXiv:2605.15763 (2026).
<a id="reference-54"></a>
54. [Retrieval Augmentation for Lay Language Generation (RALL/CELLS)](https://arxiv.org/abs/2211.03818). arXiv:2211.03818.
<a id="reference-55"></a>
55. You, et al. (2025). [PlainQAFact](https://arxiv.org/abs/2503.08890). arXiv:2503.08890.
<a id="reference-56"></a>
56. [Trie Automata for Constrained Decoding over Large Finite Sets](https://arxiv.org/abs/2608.12574). arXiv:2608.12574 (2026).
<a id="reference-57"></a>
57. [Efficient Terminology Integration for LLM-based Translation](https://aclanthology.org/2024.wmt-1.51/). WMT 2024.
<a id="reference-58"></a>
58. Jaswal (2025). [It Takes Two: Terminology-Aware Translation](https://arxiv.org/abs/2511.07461). arXiv:2511.07461.
<a id="reference-59"></a>
59. Liu, Z., et al. (2026). [Can LLMs Design High-Quality Experiments? (SCOPE/OptED)](https://arxiv.org/html/2608.03501v1). arXiv:2608.03501.
<a id="reference-60"></a>
60. Li, et al. (2025). [AutoSDT](https://arxiv.org/html/2506.08140v1), Table 3. EMNLP 2025; arXiv:2506.08140.
<a id="reference-61"></a>
61. Li, et al. (2025). [AgentExpt](https://arxiv.org/abs/2511.04921). arXiv:2511.04921.
<a id="reference-62"></a>
62. [Evaluating RAG Variants for SQL and API Call Generation](https://arxiv.org/abs/2602.07086). arXiv:2602.07086 (2026).
<a id="reference-63"></a>
63. Shaikh (2026). [Dissecting Agentic RAG: Component Ablation with a Local 7B Model](https://arxiv.org/html/2606.21553v1). arXiv:2606.21553.
<a id="reference-64"></a>
64. [Citation-Grounded Code Comprehension](https://arxiv.org/abs/2512.12117). arXiv:2512.12117 (2025).
<a id="reference-65"></a>
65. [Prompt-Based Abstention Fails Under Misleading Context](https://arxiv.org/abs/2608.22228). arXiv:2608.22228 (2026).
<a id="reference-66"></a>
66. Min, S., et al. (2023). [FActScore](https://arxiv.org/html/2305.14251v1), Table 6. EMNLP 2023; arXiv:2305.14251.
<a id="reference-67"></a>
67. Gao, T., et al. (2023). [Enabling LLMs to Generate Text with Citations (ALCE)](https://arxiv.org/abs/2305.14627). EMNLP 2023; arXiv:2305.14627.
<a id="reference-68"></a>
68. Bohnet, B., et al. (2023). [Attributed Question Answering (AutoAIS)](https://arxiv.org/abs/2212.08037). arXiv:2212.08037.
<a id="reference-69"></a>
69. [Retrieval-Augmented Agentic Rubric Generation for Medical Response Evaluation](https://arxiv.org/abs/2601.15161). arXiv:2601.15161 (2026).
<a id="reference-70"></a>
70. [When Retrieval Helps and Distracts: Biomedical Claim Verification](https://arxiv.org/abs/2608.01409). arXiv:2608.01409 (2026).
<a id="reference-71"></a>
71. [Time to REFLECT: Can We Trust LLM Judges for Evidence-based Research Agents?](https://arxiv.org/abs/2605.19196). arXiv:2605.19196 (2026).
<a id="reference-72"></a>
72. Singh, S., et al. (2024). [Global-MMLU](https://arxiv.org/abs/2412.03304). arXiv:2412.03304.
<a id="reference-73"></a>
73. [Generative vs Encoder Models for Multilingual NER](https://arxiv.org/abs/2608.29959). arXiv:2608.29959 (2026).
<a id="reference-74"></a>
74. [Copy-Paste to Mitigate LLM Hallucinations](https://arxiv.org/abs/2510.00508). arXiv:2510.00508 (2026).
<a id="reference-75"></a>
75. [Code-Switching Information Retrieval](https://arxiv.org/abs/2604.17632). arXiv:2604.17632 (2026).
<a id="reference-76"></a>
76. [The Multilingual Curse at the Retrieval Layer: Evidence from Amharic](https://arxiv.org/abs/2605.24556). arXiv:2605.24556 (2026).
<a id="reference-77"></a>
77. [Zero-Shot Cross-Lingual Biomedical Concept Normalization](https://www.medrxiv.org/content/10.1101/2025.02.27.25323007v1.full). medRxiv (2025).
<a id="reference-78"></a>
78. Yun, H. S., et al. (2024). [Extracting Numerical Results from Randomized Controlled Trials](https://arxiv.org/abs/2405.01686). arXiv:2405.01686.
<a id="reference-79"></a>
79. Ma, Z., Kim, D., & Chen, T.-H. (2024). [LibreLog](https://arxiv.org/abs/2408.01585). arXiv:2408.01585.
<a id="reference-80"></a>
80. Xiao, Y., Le, V.-H., & Zhang, H. (2024). [LogBatcher](https://arxiv.org/abs/2406.06156). arXiv:2406.06156.
<a id="reference-81"></a>
81. [Architecture Matters More Than Scale: Financial QA under SME Compute Constraints](https://arxiv.org/abs/2604.17979). arXiv:2604.17979 (2026).
<a id="reference-82"></a>
82. [LLMStructBench](https://arxiv.org/abs/2602.14743). arXiv:2602.14743 (2026).
<a id="reference-83"></a>
83. [llama.cpp](https://github.com/ggml-org/llama.cpp). CPU and GPU support.
<a id="reference-84"></a>
84. [Qwen 2.5 speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html). Measured memory examples.
<a id="reference-85"></a>
85. Jiang, et al. (2026). [Specialize Roles, Mix Deployments](https://arxiv.org/abs/2606.20629). arXiv:2606.20629.
<a id="reference-86"></a>
86. Belcak, P., et al., NVIDIA (2025). [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153). arXiv:2506.02153.
<a id="reference-87"></a>
87. [Justified or Just Convincing?](https://arxiv.org/abs/2604.04418). arXiv:2604.04418 (2026).
<a id="reference-88"></a>
88. Kenton, Z., et al., DeepMind (2024). On Scalable Oversight with Weak LLMs Judging Strong LLMs. NeurIPS 2024.
<a id="reference-89"></a>
89. Sen, S., Kasturi, Lumer, Gulati, & Subbiah (2026). [Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](https://arxiv.org/html/2605.15184v1), Table 1. arXiv:2605.15184.
<a id="reference-90"></a>
90. Panthi & Abdelfattah (2026). [Fixed RAG Compression Collapses Measured Reader Scaling](https://arxiv.org/html/2606.21807v1). arXiv:2606.21807.
<a id="reference-91"></a>
91. Kirichenko, P., et al. (2025). [AbstentionBench](https://arxiv.org/html/2506.09038v1). arXiv:2506.09038.
<a id="reference-92"></a>
92. Wei, J., et al. (2024). [Long-form Factuality in Large Language Models (SAFE)](https://arxiv.org/abs/2403.18802). NeurIPS 2024; arXiv:2403.18802.
<a id="reference-93"></a>
93. Chern, I., et al. (2023). [FacTool](https://arxiv.org/abs/2307.13528). arXiv:2307.13528.
<a id="reference-94"></a>
94. Lopez, I., et al. (2025). [CLEAR: Clinical Entity Augmented Retrieval](https://www.nature.com/articles/s41746-024-01377-1). npj Digital Medicine.
<a id="reference-95"></a>
95. [BM25 Wins at Scale](https://arxiv.org/abs/2607.26497). arXiv:2607.26497 (2026).
<a id="reference-96"></a>
96. Hsieh, C.-Y., et al. (2023). [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301). Findings of ACL 2023; arXiv:2305.02301.
<a id="reference-97"></a>
97. [Scaling Down to Scale Up](https://arxiv.org/abs/2312.14972). ISPASS 2024; arXiv:2312.14972.
<a id="reference-98"></a>
98. Anthropic. [Can I use my outputs to train an AI model?](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model) Help Center, checked 12 September 2026.
<a id="reference-99"></a>
99. Meta. [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
<a id="reference-100"></a>
100. Kandpal, N., et al. (2023). [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411). ICML 2023; arXiv:2211.08411.
<a id="reference-101"></a>
101. Mallen, A., et al. (2023). [When Not to Trust Language Models](https://arxiv.org/abs/2212.10511). ACL 2023; arXiv:2212.10511.
<a id="reference-102"></a>
102. Amiraz, C., et al. (2025). [The Distracting Effect](https://arxiv.org/abs/2505.06914). ACL 2025; arXiv:2505.06914.
<a id="reference-103"></a>
103. Liu, N. F., et al. (2024). [Lost in the Middle](https://arxiv.org/abs/2307.03172). TACL 2024; arXiv:2307.03172.
