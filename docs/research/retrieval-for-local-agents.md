# Retrieval for Local Agents: What Retrieval Can and Cannot Substitute For in the Claim Graph Pipeline

*Complete report. A short decision version is at [retrieval-for-local-agents-decision.md](retrieval-for-local-agents-decision.md). Companion to [Which Agents Go Local?](which-agents-go-local.md), which asks which roles can run on small local models at all; this report asks what retrieval adds once they do, and what it cannot replace.*

## Abstract

For the agents previously judged able to run on small local models (4–32B), can dense, lexical, hybrid or structured retrieval be combined with their work to make a local deployment more accurate or cheaper, and if retrieval does so much of the work, what is left for the model? The literature favours task-dependent retrieval over generic retrieval-augmented generation: literal-span tasks are the strongest candidates for lexical retrieval or exact lookup, while paraphrase-sensitive tasks generally require hybrid retrieval and reranking. On the second question the evidence is clear and often misread. Retrieval substitutes for one thing, the knowledge a model would otherwise have to store in its weights, and it does that well: retrieval-augmented models with tens of times fewer parameters match closed-book models on knowledge-intensive tasks. It does not substitute for reading, for resolving conflicts between retrieved evidence and prior belief, for judgement over paraphrased text, for synthesis, or for deciding when to search. Deterministic code substitutes for a different class of work, literal verification and lookup, and can remove those decisions from the model entirely. What remains for a model in every role of this pipeline is the mapping from natural-language evidence to a typed judgement, and that is where the reader's capacity binds: untuned models at 7B or below fail to use even gold evidence, and no reviewed study establishes a transition at 14B. A small model is therefore necessary wherever a role must read, sufficient only where retrieval and deterministic checks have shrunk the decision surface to something a specialist or a fine-tuned small reader can carry, and provisionally insufficient for open-ended synthesis. No study evaluates the complete pipeline end to end, so these conclusions are design hypotheses to validate per role.

## 1. Question, scope and method

**The question.** For Extraction, the Evidence Reviewer, the Critic's support and consistency audit, the Elaborator, the Experiment Designer, the Experiment Validator, the Profile Compiler, the Result Interpreter, the Reader Translator and the Translation Verifier, can other retrieval approaches be combined with the role's work to make a local deployment more accurate or cheaper? If yes, how; if no, why. And, underneath that, which part of each role's work is retrieval doing, which part is deterministic code doing, and which part still requires a model?

**Terminology.** **Retrieval** selects records from a corpus. **Lookup** addresses an exact or keyed store. **Verification** scores whether evidence supports a claim. **Orchestration** schedules, delivers and validates those operations. The report distinguishes these even when they appear in one pipeline, because the answer to the question above depends on the distinction. **Reading** or **utilisation** is the model's use of evidence placed in its context; it is a separate capability from retrieval and from the model's stored knowledge.

**Evidence labels.** **Direct** means the cited study tests the stated mechanism on substantially the same task. **Near analogue** means it tests the mechanism in a different task, domain, model tier or pipeline. **Mechanistic inference** combines separately demonstrated components. **Untested proposal** has no end-to-end evaluation. Source type is noted as peer-reviewed, preprint, blog or vendor-reported where material. Cost claims refer to the stated study's metric; marginal compute excludes index construction and maintenance unless stated; API dollars, tokens, latency, local hardware cost and energy are not treated as interchangeable.

**Method.** This is a targeted narrative evidence review. It is anchored on Sen et al.'s "Is Grep All You Need? How Agent Harnesses Reshape Agentic Search", read in full <sup>[16](#reference-16)</sup>, and draws on the linked sources listed at the end. No reproducible search protocol, screening flow or formal quality appraisal was recorded, and the review is not systematic. Section 2 adds the retrieval-augmentation and knowledge-capacity literature needed to answer the division-of-labour question; those sources are cited from their primary records.

## 2. The division of labour: what retrieval, deterministic checks and model capacity each supply

A natural reading of the per-role results in Sections 3 to 5 is that retrieval does most of the work, and that a small model is therefore an accident of deployment rather than a requirement. That reading conflates three different things that a retrieval-augmented agent does. This section separates them, states what the literature shows retrieval can substitute for, and identifies where a model is necessary and why a small one can be enough.

### 2.1 Retrieval substitutes for stored knowledge

Retrieval-augmented generation was introduced as a way to give a parametric model access to a non-parametric memory that can be inspected and updated <sup>[1](#reference-1)</sup>. The strongest results for it concern knowledge. RETRO, a 7.5B model retrieving from a two-trillion-token database, reached language-modelling performance comparable to GPT-3 and Jurassic-1 with about 25 times fewer parameters <sup>[2](#reference-2)</sup>. Atlas, an 11B retrieval-augmented model, exceeded a 540B closed-book model on NaturalQuestions with 64 training examples, with roughly fifty times fewer parameters <sup>[3](#reference-3)</sup>. Kandpal et al. showed why: a model's question-answering accuracy depends strongly on how many documents relevant to the question appeared in pretraining, the long tail of facts is learned poorly at any scale, retrieval augmentation reduces that dependence, and closing the gap through scale alone would require models many orders of magnitude larger than any built <sup>[4](#reference-4)</sup>. Mallen et al. found the same pattern on entity popularity, with retrieval helping most on rare entities and sometimes hurting on popular ones, which motivates retrieving adaptively rather than always <sup>[5](#reference-5)</sup>. Allen-Zhu and Li's capacity scaling laws give the mechanism: a sufficiently trained model stores about two bits of knowledge per parameter, so the amount of factual knowledge a model can hold is roughly linear in its size <sup>[6](#reference-6)</sup>. A small model cannot store the long tail; a corpus can. This is the precise sense in which retrieval can "fulfil a lot of what a small model does", and in this pipeline it is the sense that matters least, because the prompts were already written to minimise dependence on stored knowledge: evidence anchors are self-contained, the Synthesist may quote only in-context passages, and the Designer must ground materials in retrieval or abstain.

### 2.2 Retrieval does not substitute for reading

Once the right passage is in the context, the model still has to use it, and that capability does not come with the retriever. Pandey's study identifies severe failures to use even oracle context in untuned models at 7B and below, with utilisation failure rates of 85–100% in the reported configurations, and does not test 14B <sup>[7](#reference-7)</sup>. The PRGB benchmark reports Qwen2.5 scaling gains on retrieval-grounded tasks that vary by language and subtask, from 49.3 to 58.8 across 7B, 14B and 32B on its Chinese aggregate and from 63.16 to 66.70 on its English aggregate, so the reader's size still moves the result after retrieval is fixed <sup>[8](#reference-8)</sup>. Position matters: Lost in the Middle reports accuracy differences of 15–30 points depending on where the relevant passage sits in the context <sup>[9](#reference-9)</sup>. Distraction matters: irrelevant context sharply degrades reasoning even when the relevant information is present <sup>[10](#reference-10)</sup>, and a topically related but incorrect passage reduced accuracy by as much as 25 points in one evaluation, a loss that hard-negative retriever training did not eliminate <sup>[11](#reference-11)</sup>. The contrary finding that random documents can help is contested <sup>[12](#reference-12), [13](#reference-13)</sup>, and should not be assumed. Training the reader on retrieved context with distractors improves robustness, which is a change to the model, not the retriever <sup>[14](#reference-14)</sup>. Context compression interacts with reader capability: compression gains generally fall as the uncompressed reader's baseline rises, with no distinct capability breakpoint, and some methods harm every tested reader on particular datasets <sup>[15](#reference-15)</sup>. Finally, the anchor study shows that how evidence is delivered moves accuracy as much as which retriever produced it, and that file-based delivery collapsed one strong pairing from 93.1% to 55.2% because gains from programmatic routing are realised only when the agent reliably closes the loop <sup>[16](#reference-16)</sup>. Retrieval, in other words, delivers evidence to a reader; the reader's capacity to use it is a separate variable, and for untrained small models it is the binding one.

### 2.3 Retrieval does not resolve conflicts between evidence and memory

A model that has stored a belief and retrieves a passage that contradicts it has to decide which to trust. Surveys of knowledge conflict document that models resolve context–memory conflicts inconsistently, sometimes following the context and sometimes their prior <sup>[17](#reference-17)</sup>. Work on the interplay between parametric and contextual knowledge reports that models under-use supplied context in identifiable ways <sup>[20](#reference-20)</sup>; ParamMute proposes suppressing the model's knowledge-critical pathways to make generation more faithful to retrieved context <sup>[18](#reference-18)</sup>; and Task Matters reports that how a model resolves such a conflict depends on the knowledge requirements of the task <sup>[19](#reference-19)</sup>. The failure that matters most for this pipeline is abstention: across three frozen 3.8B–8B models, explicit abstention prompts still elicited answers on 41.6% of cases with misleading retrieved context, although the models usually abstained when evidence was simply missing <sup>[21](#reference-21)</sup>, and AbstentionBench, which is not a retrieval experiment, found little average improvement from scaling Llama 3.1 from 8B to 405B and declines for two reasoning-tuned comparisons <sup>[22](#reference-22)</sup>. This is the class of error the pipeline's deterministic verdict fusion is designed to contain: the model grades, and fixed rules decide, so a model that trusts the wrong source produces a wrong grade rather than a wrong verdict directly. The containment is partial, which is why calibration (Study B in the [Research Agenda](research-agenda.md)) is a precondition for deploying any smaller reader.

### 2.4 Deterministic checks substitute for a different class of judgement

Some of what looks like retrieval in the per-role proposals is not retrieval at all. Checking that a quoted span exists in its source, that a dataset name or a number occurs in a table, that a citation ID resolves within the supplied batch, or that a derived total equals the sum of its parts, is lookup and recomputation, and deterministic code does it with no model in the loop. The decompose-retrieve-verify-per-atom pattern of FActScore, SAFE and FacTool relies on exactly this separation between locating evidence and scoring support <sup>[23](#reference-23), [24](#reference-24), [25](#reference-25)</sup>; deterministic mediation of scientific workflows makes the same argument at the tool level <sup>[26](#reference-26)</sup>; and in one evaluation, mechanical citation-ID overlap checking reached 92% citation accuracy with zero hallucinated citations, whereas retrieval-only systems fabricated 17% of references <sup>[27](#reference-27)</sup>. The pipeline's commit validator, its quote matcher and its overclaim guard are instances of this class. The important property is that these checks remove decisions from the model entirely, rather than helping the model make them. They therefore reduce the capacity a role needs, and they are the reason a small model can be safe in roles where a literal check catches the dominant error.

### 2.5 What remains for a model, and why a small one can be enough

After retrieval has supplied the evidence and deterministic code has verified the literal properties, every role in this pipeline still has to perform one operation that neither can: read natural-language text and produce a typed judgement about it. Extraction reads a seed and emits typed nodes and edges; the Evidence Reviewer reads a passage and grades entailment and study design; the Critic reads a mechanism chain against quotes; the Validator reads a plan against its citations; the Elaborator and Translator rewrite. Retrieval does not grade, and code cannot read paraphrase. This is what makes a model necessary, and it is the reason the question is never "retrieval or a model" but "how much model, once retrieval and code have done their part".

Three lines of evidence say that a small model can be enough for narrow forms of this operation, and one says where it is not. First, purpose-built specialists match frontier models on narrow judgements: an off-the-shelf NLI evaluator reached 83.57% accuracy against GPT-4o's 83.60% on reference-based QA evaluation, and NLI plus a lexical feature reached 84.50% <sup>[28](#reference-28)</sup>; a 0.1B hallucination-detection model is reported by its vendor to exceed GPT-4-0613 on three English benchmarks by 0.17–2.64 balanced-accuracy points <sup>[29](#reference-29)</sup>; fine-tuned NLI models are competitive on scientific claim verification <sup>[30](#reference-30)</sup>; and in structured extraction a 0.3B encoder beats ChatGPT out of domain, guideline conditioning adds more than scaling from 7B to 34B, and tuned models of 3B or less beat frontier zero-shot on seven relation-extraction benchmarks <sup>[31](#reference-31), [32](#reference-32), [33](#reference-33)</sup>. The pattern requires task-specific training, and fine-tuned judges generalise poorly beyond their training tasks <sup>[34](#reference-34)</sup>, so a specialist is a candidate for one narrow judgement, not for the composite role. Second, for synthesis over supplied literature, an 8B model with a retrieval datastore matched or exceeded GPT-4o on a scientific literature-synthesis benchmark and sharply reduced citation hallucination <sup>[35](#reference-35)</sup>; this shows that retrieval plus a small reader can carry grounded synthesis, and it does not test novel hypothesis generation, which is why the Synthesist stays on a frontier model provisionally. Third, for orchestration, small models can be trained to operate search loops: reinforcement learning teaches 3B and 7B models to interleave reasoning with search calls and improves substantially over retrieval-augmented baselines <sup>[36](#reference-36), [37](#reference-37)</sup>, while an untrained 7B model does about as well with a fixed two-step schedule as with five steps, and a fixed hybrid retriever beat its heuristic selector <sup>[38](#reference-38)</sup>. The choice is therefore between a deterministic schedule and a trained search agent, not between retrieval and no retrieval.

Where the evidence says a small model is not enough is the reading constraint of Section 2.2: untuned readers at 7B and below can fail to use the evidence retrieval has found, and no reviewed study establishes that the failure disappears at 14B. Retrieval cannot fix that, because it is upstream of it. The remedies are a larger reader, a trained reader, or a narrower decision surface, and which one is cheapest is an empirical question per role.

Why a small model rather than a frontier model for the reading function, when the frontier model can also read? Cost per accepted result, since the Evidence Reviewer makes about two-thirds of core calls and the Synthesist holds the deepest context; latency and concurrency for the high-volume roles; privacy for the Result Interpreter, whose inputs are the researcher's own unpublished results; and the routing economics that make heterogeneous deployments attractive, in which cascades and learned routers cut cost substantially at small quality loss by sending easy queries to small models and hard ones to large models <sup>[39](#reference-39), [40](#reference-40), [41](#reference-41)</sup>. The NVIDIA position paper argues on these grounds that small models are sufficiently powerful, more suitable and more economical for many agentic invocations, and estimates that a large share of calls in typical agents could move to them; that share is an expert estimate, not a measurement, and the one rigorous role-assignment study reviewed in the companion report found that mis-assigning the strong model swings accuracy by tens of points in either direction <sup>[42](#reference-42)</sup>. On-premise deployment carries its own costs and trade-offs <sup>[43](#reference-43)</sup>. None of this is a reason to prefer a small model where it cannot read; it is a reason to measure whether it can.

### 2.6 The division of labour by role

| Role | Knowledge access | Literal verification | Reading and judgement (model required) | Generation | Orchestration |
|---|---|---|---|---|---|
| Extraction | Retrieved demonstrations, schema slices, alias tables | Verbatim-span check, schema validity, dedup shortlist | Map seed text to typed nodes and edges; specialist encoder or tuned 7–9B | None beyond the typed output | Single inline prompt |
| Evidence Reviewer | Hybrid retrieval, reranking, passage budget | Quote existence; role requirements enforced by the commit validator | Entailment and study-design grading over paraphrased passages; NLI specialist for entailment, 14–32B reader for the rest | None | Fixed schedule; verdict by deterministic fusion |
| Critic | Retrieve and rerank prior work | Quote existence | Mechanism and terminology audit (specialist plus small reader); novelty synthesis (frontier provisionally) | Critique text | Fixed listwise call; escalation rule |
| Experiment Designer | Curated, entity-linked methods index | Citation-ID enforcement; abstention field | Turn a mechanism into a design; 32B start, 14B candidate | Plan text | Short deterministic schedule |
| Experiment Validator | Retrieve supporting passages | Lexical pre-filter for names and numbers | Claim–passage support scoring; 8–32B judge or NLI scorer | Feedback text | Retrieve then verify |
| Elaborator | Definition retrieval | Verify only elaborated sentences | Rewriting that preserves meaning; about 7B | Reader card | Single call |
| Reader Translator | Glossary lookup | Constrained decoding over the finite set | Rewriting; 7–13B translation models | Rewrite | Single call |
| Translation Verifier | None identified | Segment alignment | Meaning-change detection; quality-estimation models | None | Single call |
| Profile Compiler | Glossary and terminology lookup; lexical retrieval for code-switched text | Span grounding | Multilingual extraction; fine-tuned extractor only | Typed profile | Single call plus user confirmation |
| Result Interpreter | Template caches, structured lookup | Source coordinates, numeric gates, recomputation | Map table or log to typed evidence; code-specialised small model behind gates | Typed evidence | Cache-first, then model |

Read across a row, the third column is the answer to "what makes the model necessary": it is the only column that requires reading paraphrased text into a typed judgement. Read down the column, the size evidence is what Sections 3 to 5 supply, and it is near-analogue everywhere.

### 2.7 What would change the answer

The division above is a hypothesis with clear failure conditions. If Pandey-style utilisation failures persist at 14B on this pipeline's own tasks, then retrieval investment cannot rescue small readers for the Evidence Reviewer and the Validator, and the roadmap should route those roles to larger readers or to trained readers. If a fine-tuned small reader matches the frontier backend on independently adjudicated edges at matched abstention, then retrieval plus a small reader is sufficient for the Evidence role and the cost argument carries. If a trained search agent at 7B beats the fixed two-step schedule on the Designer's methods retrieval, then model-driven orchestration becomes viable for untuned-scale models; if it does not, the deterministic schedule stays. And if the Synthesist comparison in the companion report shows a specialised 4–32B model matching the frontier backend on hypothesis quality with equivalent evidence access, the one role now held on a frontier model would move too. Each of these is a measurement the companion report's harness can take.

## 3. The anchor study's framework

**Evidence: direct for the tested LongMemEval-S setup (preprint).** Sen et al. compare grep-style lexical and vector retrieval inside agentic loops across four harnesses and five models on a 116-question LongMemEval subset <sup>[16](#reference-16)</sup>. Four results matter here. Inline grep beats inline vector retrieval for every tested harness–model pair, for example 93.1% versus 83.6% for Opus 4.6 on the custom harness; the largest margin is Gemini 3.1 Flash-Lite under Chronos at 86.2% versus 62.9%, a 23.3-point gap, while Claude Haiku 4.5 under Claude Code shows 55.2% versus 44.0%, an 11.2-point gap. The authors hypothesise that weaker agents are less consistent at iterative query refinement and result consumption, but the experiment does not establish a monotonic capability effect. Delivery mode can invert the ranking: file-based or programmatic delivery, where the agent must run extra tool steps to read results, collapsed one strong pairing from 93.1% to 55.2%. The result is scoped to verbatim-span tasks; the authors expect dense or hybrid retrieval to matter more where evidence is rarely literal, as in scientific synthesis over paraphrased abstracts. And retrieval in agents is retrieval plus orchestration, since harness and delivery choices moved accuracy as much as the retriever did.

For this pipeline the framework suggests a candidate routing rule. Tasks whose evidence is a literal span (quote existence, alias lookup, glossary substitution, dataset and metric names, numeric cells) are candidates for lexical or exact-match machinery. Tasks whose evidence is paraphrased (entailment grading, methods grounding) are candidates for hybrid retrieval with reranking, with context utilisation tested separately. Inline delivery is a strong default for untrained small models on similar tasks; multi-step file-based tool loops should be used only after direct validation or task-specific training.

**Evidence: near analogue (peer-reviewed, preprint and blog sources).** Several heterogeneous studies provide convergent, non-replicative support for lexical or entity-anchored retrieval on literal evidence. CLEAR obtained 0.90 average F1 versus 0.86 for embedding retrieval in clinical extraction, with 72% lower latency and 71% fewer input tokens <sup>[44](#reference-44)</sup>. One enterprise-QA benchmark found BM25's advantage over dense retrieval widening as its corpus grew, from 74.7% against 58.1% to 50.5% against 29.9% <sup>[45](#reference-45)</sup>, and a non-peer-reviewed synthetic entity-resolution experiment found BM25 strongest on pristine lexical matches while fine-tuned dense models helped corrupted inputs <sup>[46](#reference-46)</sup>. Because these tasks and pipelines differ from the anchor experiment, they are analogues rather than independent replications, corpus size alone should not be treated as a universal reason to upweight lexical retrieval, and literal fidelity is not guaranteed by parameter count.

## 4. Cross-cutting results

Findings that apply to several agents at once, all near analogues.

- **Long-tail question answering can make retrieval load-bearing.** Across the cited settings, closed-book accuracy for 7B-class models rose from 5–12% to 44–56% with one gold passage, and retrieval-augmented small models outperformed much larger parametric-only models on tail entities <sup>[4](#reference-4), [5](#reference-5)</sup>. Applicability to this causal-claim pipeline remains a hypothesis to test, and Section 2.1 explains why it matters less here than in open-domain QA.
- **Passage curation can outweigh retriever swaps.** A topically related but incorrect distractor reduced accuracy by as much as 25 points, and hard-negative retriever training did not eliminate the loss <sup>[11](#reference-11)</sup>. Reranking or filtering before the model call is therefore a candidate intervention.
- **At the smallest tested scales, context utilisation can bind before retrieval quality.** Pandey identifies severe oracle-context utilisation failures for untuned models at 7B and below without establishing a 14B transition <sup>[7](#reference-7)</sup>; PRGB's scaling gains vary by language and subtask <sup>[8](#reference-8)</sup>. Treat 14B as a candidate deployment tier requiring direct in-domain comparison with 7B and 32B.
- **Validate compression per reader, dataset and method.** Compression gain generally decreases as the uncompressed reader's baseline improves, with correlations near −0.9 for several compressor and dataset pairs, but the study finds no distinct capability breakpoint; do not enable or disable compression using a fixed threshold <sup>[15](#reference-15)</sup>.
- **Misleading retrieved context can impair abstention.** Explicit abstention prompts still elicited answers on 41.6% of misleading cases in three frozen 3.8B–8B models <sup>[21](#reference-21)</sup>; AbstentionBench's scaling results are not retrieval experiments <sup>[22](#reference-22)</sup>. Use an explicit abstention field together with evidence-conflict checks, and validate missing and misleading conditions separately.
- **Passage placement can matter.** One long-context study reported 15–30 accuracy-point differences associated with passage position; test best-first and best-last ordering rather than placing the most relevant evidence in the middle <sup>[9](#reference-9)</sup>.
- **Decompose, retrieve and verify per atom is an established baseline pattern.** FActScore, SAFE and FacTool demonstrate variants of it for factual verification <sup>[23](#reference-23), [24](#reference-24), [25](#reference-25)</sup>; it resembles the pipeline's Evidence and Validator design without validating that composite design end to end.

## 5. Agent-by-agent analysis

### 5.1 Extraction: yes

Four candidate channels follow; the evidence is direct only for each cited source task and near analogue for this pipeline.

1. **Retrieved demonstrations** instead of a static few-shot set. On two relation-extraction evaluations, contrastive demonstration retrieval increased Llama-3.1-8B F1 from 15.7 to 38.7 and from 8.3 to 21.5, relative gains of about 146% and 159% from low baselines; in the same study the relative gains at 70B were about 48–51% <sup>[47](#reference-47)</sup>. GPT-RE established the retrieved-demonstration pattern for relation extraction <sup>[48](#reference-48)</sup>, and in separate work five well-retrieved demonstrations scored 83.4 F1 versus 80.3 for thirty generically retrieved ones.
2. **Schema-slice retrieval**, narrowing the closed schema shown to the model instead of dumping the whole type system in context. A trained schema retriever raised a 1B model's F1 from 0.23 with a BM25 baseline to 0.42 and reported 24–52% latency savings from caching <sup>[49](#reference-49)</sup>; a separate frontier-scale study reported an F1 gain with roughly halved latency and tokens <sup>[50](#reference-50)</sup>. The 4–14B range was not tested in those results. For this pipeline's fixed ten-type vocabulary, use deterministic dictionary lookup to inject the relevant type guidelines; guideline conditioning added 13 F1 points in a 7B result <sup>[32](#reference-32)</sup>.
3. **Verbatim-span grounding with deterministic verification.** A filings-extraction system required at least 40% of a quote's four-word runs to appear verbatim in the source and retried otherwise; its source-reported unsupported-claim rate fell from 8% to approximately 0% <sup>[51](#reference-51)</sup>. A schema-grounded clinical extractor with validator-in-the-loop repair raised structural validity from 0.51 to 0.96 in a median of 1.2 repair rounds <sup>[52](#reference-52)</sup>. Neither study measures the recall cost of forced verbatim spans, so track true claims lost for lack of clean contiguous support.
4. **Node deduplication and canonicalisation against the existing graph.** A candidate cascade is exact or alias-table lookup, BM25 blocking to a shortlist of about five to ten, an optional compact bi-encoder such as SapBERT for paraphrase or cross-lingual variants <sup>[53](#reference-53)</sup>, then small-model adjudication over the shortlist. Fine-tuned 7–13B matchers met or beat GPT-4 on most evaluated entity-matching benchmarks <sup>[54](#reference-54)</sup>; a 124M model came within 4.4 entity-matching F1 points of GPT-4 at a source-reported inference cost thousands of times lower under that paper's assumptions <sup>[55](#reference-55)</sup>; locally served Qwen3-4B and 8B gained 2.6–24.2 F1 points from retrieval augmentation <sup>[56](#reference-56)</sup>; and bounded shortlist selection added 19–32 points at 7–8B in a separate study <sup>[57](#reference-57)</sup>.

**How it fits.** All four channels are single-shot, deterministic-core-driven lookups assembled into one inline prompt. This avoids an open-ended model-driven search loop and follows the delivery pattern favoured by the anchor study, although that study did not test open-weight 4–32B models. In the terms of Section 2, the channels supply knowledge (demonstrations, schema, aliases) and literal verification (spans, dedup); the model still reads the seed.

### 5.2 Evidence Reviewer: yes, with a documented ceiling

The retrieval-and-delivery half of this agent's pipeline has four candidate upgrades, all near analogues, ordered along the existing enrichment, triage and review stages.

1. **Hybrid retrieval plus a small reranker upstream.** Add a BM25 channel beside the dense channel, fuse with reciprocal-rank fusion, and rerank with a small cross-encoder before the triage cutoff. On a scientific-claim retrieval task, hybrid plus reranking reached Recall@5 of 0.816 versus 0.587 for dense-only <sup>[58](#reference-58)</sup>. Monitor corpus-size effects rather than assuming a universal direction <sup>[45](#reference-45)</sup>.
2. **Moderate context pruning between triage and the prompt.** Extractive pruning was accuracy-neutral or positive in the cited evaluations, and abstractive compression to 5–10% of the original tokens incurred under 10% relative accuracy loss <sup>[59](#reference-59), [60](#reference-60)</sup>. Because this agent grades five fine-grained axes, validate reader, dataset and compressor together and cap compression where those axes degrade. Test best-first and best-last ordering.
3. **A lightweight NLI offload for the entailment map.** The DIVER-QA result in Section 2.5 supports testing a purpose-built classifier to populate or cross-check the entailment and contradiction-risk fields per edge–passage pair <sup>[28](#reference-28)</sup>; it does not establish that small NLI models generally outperform frontier models on long-context entailment.
4. **An adaptive passage budget.** Activate the specification's currently unused budget and scheduling helpers rather than reviewing every triaged candidate by default, since a smaller passage budget can beat a larger one when the additional passages are near-miss distractors <sup>[11](#reference-11)</sup>.

**The model-side constraint.** For untuned models at 7B and below, context utilisation rather than retrieval binds (Section 2.2) <sup>[7](#reference-7)</sup>; the study does not test 14B or establish how much of the achievable gap retrieval investment can recover. Treat 14B–32B as candidate tiers for direct in-domain comparison and combine them with structural offloading rather than assuming a capability threshold. Knowledge-conflict behaviour does not reliably improve with scale in the cited work, so deterministic verdict fusion is a mitigation to test. The anchor study's scope does not support grep-only as the default for this agent's paraphrase-tolerant grading; benchmark hybrid retrieval with lexical retrieval as a complementary channel for named concepts.

### 5.3 Critic: support and consistency audit yes; novelty assessment remains frontier provisionally

**Support and consistency audit.** Because the evidence is already in context, the candidate mechanism is a two-stage verification cascade rather than corpus search: a deterministic exact or fuzzy substring check that each quoted span exists in the source, with negligible marginal compute relative to the model call, then a small specialised entailment model judging whether the audited step is semantically supported. The vendor-reported HHEM result <sup>[29](#reference-29)</sup> makes it a low-compute verifier candidate requiring in-domain validation; separately, a fine-tuned 770M model outperformed zero-shot GPT-3.5 with chain-of-thought on attribution judgment, while out-of-domain attribution accuracy for generically prompted small models fell below 60% <sup>[61](#reference-61)</sup>. These gains depend on task-specific fine-tuning or purpose-built checkpoints.

**Novelty assessment.** A retrieve-then-rerank novelty checker reported approximately 13% higher agreement with its reference judgments <sup>[62](#reference-62)</sup>, and a literature-grounded review agent reported an evidence-based-critique score of 1.48 versus 0.10 for GPT-4o <sup>[63](#reference-63)</sup>. The reviewed systems routed cross-paper comparative synthesis through 70–120B-class models, with at most a 14B model drafting. This is paraphrased, conceptual, cross-document evidence outside the anchor study's grep result. Retain the frontier synthesis tier provisionally because no direct small-model result was found; that absence is not evidence that small-model synthesis is impossible.

### 5.4 Experiment Designer: curated retrieval supported; unrestricted search not supported

The reviewed studies show a task-specific split.

- **Free-text search inside the planning chain of thought did not help most tested models.** Direct for the SCOPE experiment-design benchmark (preprint). Adding chain-of-thought plus web search produced no significant gain for five of seven models, lowered one flagship model's total score from 18.22 to 16.77, and increased DeepSeek-V3.2's aggregate redline rate from 7.67% to 14.00%; the full OptED workflow reduced that rate to 2.67% and the stage-isolation-only ablation reached 4.33%. Redline rate combines source hallucinations, metric incompatibilities and constraint violations, so it should not be read solely as fabrication, and OptED combines several workflow changes, so the study does not isolate retrieval as the dominant lever over model capability <sup>[64](#reference-64)</sup>.
- **Structured, curated retrieval improved recommendation.** Direct for AgentExpt recommendation and near analogue for planning (preprints). AgentExpt builds its knowledge base from 108,825 papers across ten AI venues, with auxiliary metadata from Papers with Code, OpenAlex, Hugging Face and Kaggle; its fine-tuned 0.6B retriever and 7B reranker outperformed graph-aware SymTax by 4.46–7.23% relative on Recall@20 and 7.52–8.27% on HitRate@10 depending on the task <sup>[65](#reference-65)</sup>. In a separate structured-target study, grounding raised execution accuracy from 0% exact match without grounding to 71–79% on SQL and API generation <sup>[66](#reference-66)</sup>.
- **Use a short deterministic schedule as the initial default for untrained small models.** On a 5,000-question HotpotQA sample with Qwen2.5-7B, fixed hybrid reciprocal-rank fusion outperformed a rule-based retrieval-strategy selector by 1.8 exact-match points, and two fixed retrieval iterations reached 52.9 exact match versus 53.2 at five steps <sup>[38](#reference-38)</sup>. The study does not test a model-driven router, and trained small models can operate search loops <sup>[36](#reference-36), [37](#reference-37)</sup>, so these results do not support a universal prohibition.
- **Enforce citation IDs mechanically and give abstention a schema branch.** Mechanical citation-ID overlap checking reached 92% citation accuracy with zero hallucinated citations in the cited evaluation, whereas retrieval-only systems fabricated 17% of references <sup>[27](#reference-27)</sup>. Because misleading retrieved context can impair abstention <sup>[21](#reference-21)</sup>, test an enforced "no passage found" output field together with evidence-conflict checks under both missing and misleading evidence.

**How it fits.** Test a deterministic pre-retrieval stage over curated structured sources feeding the 32B planner inline, with retrieval isolated in its own schema-constrained step and the Validator checking what enforcement misses. The source results are not additive and are study-specific reference points, not expected effect sizes for this pipeline.

### 5.5 Experiment Validator: yes

**Evidence: near analogue (peer-reviewed and preprint sources).** The citation-support check follows a retrieve-then-verify pattern. In FActScore's biography evaluation, retrieval reduced the ChatGPT evaluator's aggregate estimation error from 39.6 to 5.1 points on InstructGPT generations, and an instruction-tuned 7B estimator obtained aggregate errors of 1.4, 0.4 and 9.9 points across three generators against 5.2, 4.7 and 8.7 for ChatGPT; these are aggregate estimation errors, not per-claim verification-error rates, and individual verification quality must be described using the paper's micro-F1 results <sup>[23](#reference-23)</sup>. The mechanism to test is the ALCE and AutoAIS pattern: a fixed-purpose NLI scorer over cited-passage and claimed-dataset pairs, with a deterministic lexical pre-filter for literal dataset names and numbers before any model call <sup>[67](#reference-67), [68](#reference-68)</sup>. A separate study reported 18–44 accuracy-point gains in criteria alignment with expert gold standards when grading criteria were grounded in retrieved domain literature <sup>[69](#reference-69)</sup>. Two caveats apply: retrieval utility varied from +0.68 to −0.38 with corpus–claim alignment in a biomedical claim-verification study, so do not add an unvalidated second open-domain search pass <sup>[70](#reference-70)</sup>; and retrieval-augmented small judges remained less reliable than large ones in the cited evaluation of judges for evidence-based research agents <sup>[71](#reference-71)</sup>. The deterministic core should remain the sole component permitted to accept and persist changes.

### 5.6 Elaborator: yes

**Evidence: near analogue (peer-reviewed study and preprint).** Two placements are supported on source tasks. Conditioning lay rewriting on retrieved term definitions improved readability and factual correctness over a no-retrieval baseline on the 63,000-pair CELLS corpus <sup>[72](#reference-72)</sup>. Classifying output sentences as simplification or elaboration and routing only elaboration sentences through a retrieval-augmented QA check increased ROC-AUC by 8.3 points, from 0.727 to 0.810, relative to its no-retrieval configuration <sup>[73](#reference-73)</sup>. The second design concentrates retrieval on the sentences at greater risk of fabricated explanation; its transfer to this pipeline is untested.

### 5.7 Reader Translator: yes

Reader-vocabulary substitution is a finite-set, verbatim-span operation. **Evidence: mechanistic inference; the composite pipeline is untested (preprints).** Retrieve candidate glossary entries by exact-match lookup rather than vector search, then use trie-based constrained decoding for the substitution field; this restricts that field to retrieved terms when the substitution is emitted through the constraint, but it does not constrain surrounding free text, and the cited trie-automata study demonstrates finite-set validity and latency behaviour rather than a task-performance gain <sup>[74](#reference-74)</sup>. Separate studies report improved translation consistency from glossary or terminology retrieval <sup>[75](#reference-75), [76](#reference-76)</sup> and show dense retrieval outperforming parametric generation for cross-lingual concept normalisation <sup>[77](#reference-77)</sup>. No reviewed paper evaluates glossary lookup plus constrained decoding end to end.

### 5.8 Translation Verifier: partial, mostly no

**Evidence: no direct retrieval result; the quality-estimation comparator is a preprint.** Retrieval's payoff for this role remains unknown. On WMT25, Gemma-3-27B and Qwen3-VL-30B doing plain single-pass quality estimation reached system-level soft pairwise accuracy of 0.82 and 0.83, above reference-based COMET22 at 0.73 and reference-free COMETKiwi22 at 0.60, but below GEMBA-MQM V2 at 0.84 and Gemini-2.5-Pro at 0.87; the open models remained weaker at segment-level ranking and error-span detection, and for Czech→German the best open filtered span F1 was 10.37 versus Gemini-3-Flash's 18.17 <sup>[78](#reference-78)</sup>. CompactQE does not evaluate retrieval, and no reviewed work directly tests retrieval-augmented verification for this gap; the nearest analogue, PlainQAFact's retrieval pattern, is from a different domain <sup>[73](#reference-73)</sup>. The quality-estimation literature's structural blind spot, meaning-altering omissions, is task-architectural and is not something retrieval addresses on current evidence. Do not build retrieval infrastructure here without a pilot; if the segment-level gap matters, test the retrieve-and-QA pattern experimentally.

### 5.9 Profile Compiler: partial

Three sub-answers with different verdicts.

- **Span grounding: supported on source tasks.** The verbatim-quote-plus-deterministic-validator pattern transfers from Extraction (§5.1). The cited systems reported approximately 90% verbatim-match rates and the rise in schema validity from 0.51 to 0.96 with validator-in-the-loop repair <sup>[52](#reference-52)</sup>; copy-constrained decoding plus a 365-example preference-tuning pass raised Llama-3-8B contextual faithfulness to 92.8%, compared with GPT-4o's 47.5% in the same evaluation regime <sup>[79](#reference-79)</sup>.
- **Lexical retrieval is more robust than the tested dense baselines on code-switched text.** Generic multilingual dense embedders degraded more than BM25 under code-switching (−11.9 versus −6.6 nDCG), one reranker fell by 34 nDCG points, and zero-shot multilingual dense retrievers lost to BM25 on low-resource morphologically rich languages <sup>[80](#reference-80), [81](#reference-81)</sup>. Use exact or substring lookup for within-document narrative search; if cross-document retrieval is needed, benchmark a hybrid sparse-plus-dense model rather than assuming either channel wins universally.
- **Terminology normalisation should favour retrieval over parametric recall.** A dense cross-lingual retriever outperformed the best evaluated generative model at zero-shot biomedical concept normalisation across ten languages <sup>[77](#reference-77)</sup>. Test glossary or dictionary lookup that supplies a canonical term to the model.

The composition risk of chaining normalisation, extraction and grounding on one small multilingual model is unmeasured anywhere; the in-house benchmark recommendation from the companion report stands.

### 5.10 Result Interpreter: yes, mainly through lookup and scaffolding

The strongest-supported mechanisms for this role are deterministic lookup and parsing rather than retrieval in the usual sense: a value's correct location is a literal span in the user's own table or log.

- **Exact-match template caches before model calls.** LibreLog reports that clustering and cache-matching scaffolding improved an 8B model's parsing accuracy by 13.7% relative to prior GPT-3.5- and GPT-4-backed log parsers while running 2.7 times as fast <sup>[82](#reference-82)</sup>; LogBatcher reports that a cache-first design reduced the number of model invocations by roughly half <sup>[83](#reference-83)</sup>.
- **Route retrieval architecture by task shape.** With the same 8B backbone, schema-based structured memory exceeded plain retrieval by 11.8 accuracy points on single-turn deterministic fact extraction, plain retrieval performed better on multi-turn aggregation, and task-aware routing added 2.9 points over either fixed choice <sup>[84](#reference-84)</sup>.
- **Gate every number deterministically and preserve its provenance.** In the cited structured-extraction evaluation, wrong values accounted for approximately 97% of remaining errors after structural errors were removed, and one 12B model outperformed several 70B models <sup>[85](#reference-85)</sup>. Attach each extracted number to source coordinates (document, table, row, column, labels, unit) and validate the value and labels jointly; recompute derived values deterministically; treat plain string presence as a first-pass check, not semantic validation.

This supports and sharpens the companion report's verdict of local only behind numeric gates: the gates are not merely a safety net. In the cited source tasks, the cache and structured-lookup layer also produced the measured efficiency gains of fewer model invocations and faster parsing, which is Section 2.4's point that deterministic checks remove decisions from the model rather than helping it make them.

## 6. Design rules distilled

1. **Route by evidence type.** Verbatim-span work goes to inline lexical or exact-match lookup; paraphrase work goes to hybrid BM25-plus-dense retrieval with a small reranker; finite-set work (glossaries, schemas, aliases) goes to deterministic dictionary lookup that injects the entries relevant to the current item.
2. **Deliver retrieval inline to untrained small models by default.** Use multi-step file-based or model-driven loops only after direct validation or task-specific training; trained small models can operate search loops.
3. **Test offloading sub-tasks to small specialists.** A lightweight NLI evaluator can match a frontier model on particular tasks; similar specialists include small rerankers, span locators and entity linkers. These results support a cascade that removes narrow judgements from the main model, and do not establish that every small classifier beats larger generative models.
4. **Shortlist, then adjudicate.** For matching and deduplication, retrieve a candidate set of about five to ten before model adjudication; validate the candidate-set size and effect in this pipeline.
5. **Verify spans and numbers deterministically, and make abstention a schema branch.** String-match quoted spans; for numbers, validate source coordinates, labels, units and conditions jointly and recompute derived values; require an explicit "nothing retrieved" field plus evidence-conflict checks; validate missing and misleading conditions separately.
6. **Curate and cap passages, validate compression per reader and dataset, and put evidence for the current claim first.** One near-miss distractor can cost more than a retriever swap gains, and compression gains decline as reader capability rises without a fixed breakpoint.
7. **Benchmark 7B, 14B and 32B directly for retrieval-heavy grading.** Severe oracle-context utilisation failures are documented at 7B and below, and no reviewed study establishes a transition at 14B; scaling effects vary by language and subtask.
8. **Give the deterministic core final acceptance authority.** Across the reviewed systems, validators, caches, string checks and fixed schedules often supplied the important reliability controls, and adding unrestricted search to model judgement did not improve most tested planners.
9. **Ask, for every role, which column of Section 2.6 a proposed change addresses.** A retriever upgrade addresses knowledge access; a validator addresses literal verification; only a change to the reader, its training or its size addresses reading and judgement. Mis-assigning an intervention to the wrong column is the most common way to expect a gain that retrieval cannot deliver.

## 7. Limitations

No study tests any of these agents' exact composite tasks end to end at 4–32B; most numerical results are near-analogue transfers identified by the evidence labels above. The anchor study tests frontier proprietary models rather than open 7–32B weights and presents its weak-model mechanism as a plausible hypothesis scoped to literal spans; transfer to this pipeline's local tier is analogical rather than direct measurement or independent replication. Several cited sources are preprints, vendor reports or blog posts, and some findings were reviewed only at abstract or search-synthesis level. The division-of-labour argument in Section 2 rests on well-replicated results for open-domain knowledge tasks (retrieval substituting for parameters) and on a smaller set of 2025–2026 preprints for the utilisation and conflict constraints; the former transfer to this pipeline only in weakened form, because its prompts already minimise dependence on stored knowledge, and the latter have not been tested on scientific appraisal. This was a targeted narrative evidence review: it did not record a reproducible search strategy, screening flow or formal quality appraisal. Domain transfer remains a recurring caveat: entity-matching evidence comes from e-commerce and bibliographic benchmarks, span-grounding evidence from clinical and financial extraction, and none from claim-graph concept vocabularies. The recommended in-house measurements from the companion report, calibration replay, verbatim-fidelity curve and role-swap ablation, apply unchanged to the retrieval configurations proposed here.

## 8. Conclusion

Retrieval can fulfil a great deal of what a small model would otherwise do, but only one kind of thing: it substitutes for knowledge the model would have to store, and the strongest results in the field are results about that substitution. It does not read, it does not resolve conflicts between what it finds and what the model believes, it does not grade paraphrase, it does not synthesise, and it does not decide when to search. Deterministic code takes a second class of work off the model, literal verification and lookup, and that is what makes small models safe in roles where a literal check catches the dominant error. What remains is the reading of natural-language evidence into a typed judgement, and that is the reason every role in this pipeline still needs a model. Whether a small one suffices is a question about the reader, not the retriever: specialists and fine-tuned small readers match frontier models on narrow judgements, retrieval plus an 8B reader carries grounded synthesis, trained small models can run search loops, and untuned readers at 7B and below can fail to use what retrieval finds. The measurements that decide it are the ones the companion report already proposes, taken role by role, with each intervention assigned to the column of work it actually changes.

## References
<a id="reference-1"></a>
1. Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS 2020; arXiv:2005.11401.
<a id="reference-2"></a>
2. Borgeaud, S., et al. (2022). [Improving language models by retrieving from trillions of tokens (RETRO)](https://arxiv.org/abs/2112.04426). ICML 2022; arXiv:2112.04426.
<a id="reference-3"></a>
3. Izacard, G., et al. (2023). [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299). JMLR; arXiv:2208.03299.
<a id="reference-4"></a>
4. Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411). ICML 2023; arXiv:2211.08411.
<a id="reference-5"></a>
5. Mallen, A., et al. (2023). [When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://arxiv.org/abs/2212.10511). ACL 2023; arXiv:2212.10511.
<a id="reference-6"></a>
6. Allen-Zhu, Z., & Li, Y. (2024). [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405). ICLR 2025; arXiv:2404.05405.
<a id="reference-7"></a>
7. Pandey (2026). [Can Small Language Models Use What They Retrieve?](https://arxiv.org/abs/2603.11513). arXiv:2603.11513.
<a id="reference-8"></a>
8. Tan, et al. (2025). [PRGB Benchmark](https://arxiv.org/abs/2507.22927). arXiv:2507.22927.
<a id="reference-9"></a>
9. Liu, N. F., et al. (2024). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172). TACL 2024; arXiv:2307.03172.
<a id="reference-10"></a>
10. Shi, F., et al. (2023). [Large Language Models Can Be Easily Distracted by Irrelevant Context](https://arxiv.org/abs/2302.00093). ICML 2023; arXiv:2302.00093.
<a id="reference-11"></a>
11. Amiraz, C., et al. (2025). [The Distracting Effect: Understanding Irrelevant Passages in RAG](https://arxiv.org/abs/2505.06914). ACL 2025; arXiv:2505.06914.
<a id="reference-12"></a>
12. Cuconasu, F., et al. (2024). [The Power of Noise: Redefining Retrieval for RAG Systems](https://arxiv.org/abs/2401.14887). SIGIR 2024; arXiv:2401.14887.
<a id="reference-13"></a>
13. [The Powerless Noise](https://arxiv.org/abs/2607.03615). arXiv:2607.03615 (2026).
<a id="reference-14"></a>
14. Zhang, T., et al. (2024). [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131). arXiv:2403.10131.
<a id="reference-15"></a>
15. Panthi & Abdelfattah (2026). [Fixed RAG Compression Collapses Measured Reader Scaling](https://arxiv.org/abs/2606.21807). arXiv:2606.21807.
<a id="reference-16"></a>
16. Sen, S., Kasturi, Lumer, Gulati, & Subbiah (2026). [Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](https://arxiv.org/abs/2605.15184). arXiv:2605.15184.
<a id="reference-17"></a>
17. Xu, R., et al. (2024). [Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319). EMNLP 2024; arXiv:2403.08319.
<a id="reference-18"></a>
18. [ParamMute: Suppressing Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation](https://arxiv.org/abs/2502.15543). arXiv:2502.15543 (2025).
<a id="reference-19"></a>
19. [Task Matters: Knowledge Requirements and Context-Memory Conflict in LLMs](https://arxiv.org/abs/2506.06485). arXiv:2506.06485 (2025).
<a id="reference-20"></a>
20. [Understanding the Interplay between Parametric and Contextual Knowledge for Large Language Models](https://arxiv.org/abs/2410.08414). arXiv:2410.08414 (2024).
<a id="reference-21"></a>
21. [Prompt-Based Abstention Fails Under Misleading Context](https://arxiv.org/abs/2608.22228). arXiv:2608.22228 (2026).
<a id="reference-22"></a>
22. Kirichenko, P., et al. (2025). [AbstentionBench](https://arxiv.org/abs/2506.09038). arXiv:2506.09038.
<a id="reference-23"></a>
23. Min, S., et al. (2023). [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251). EMNLP 2023; arXiv:2305.14251.
<a id="reference-24"></a>
24. Wei, J., et al. (2024). [Long-form Factuality in Large Language Models (SAFE)](https://arxiv.org/abs/2403.18802). NeurIPS 2024; arXiv:2403.18802.
<a id="reference-25"></a>
25. Chern, I., et al. (2023). [FacTool: Factuality Detection in Generative AI](https://arxiv.org/abs/2307.13528). arXiv:2307.13528.
<a id="reference-26"></a>
26. Adamidis, P., et al. (2026). [It's Not the Language Model, It's the Tool: Deterministic Mediation for Scientific Workflows](https://arxiv.org/abs/2605.13245). arXiv:2605.13245.
<a id="reference-27"></a>
27. [Citation-Grounded Code Comprehension](https://arxiv.org/abs/2512.12117). arXiv:2512.12117 (2025).
<a id="reference-28"></a>
28. Balamurali & Cheng (2025). [Revisiting NLI: Cost-Effective Metrics for QA Evaluation](https://arxiv.org/abs/2511.07659). arXiv:2511.07659.
<a id="reference-29"></a>
29. Vectara (2024–2025). [HHEM-2.1-Open](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model). Vendor blog.
<a id="reference-30"></a>
30. Košprdić, M., et al. (2024). [Scientific Claim Verification with Fine-Tuned NLI Models](https://www.scitepress.org/Papers/2024/129000/129000.pdf). SciTePress.
<a id="reference-31"></a>
31. Zaratiana, U., et al. (2024). [GLiNER: Generalist Model for Named Entity Recognition](https://arxiv.org/abs/2311.08526). NAACL 2024; arXiv:2311.08526.
<a id="reference-32"></a>
32. Sainz, O., et al. (2024). [GoLLIE: Annotation Guidelines Improve Zero-Shot Information Extraction](https://arxiv.org/abs/2310.03668). ICLR 2024; arXiv:2310.03668.
<a id="reference-33"></a>
33. [Sub-Billion, Super-Frontier](https://arxiv.org/abs/2606.22606). arXiv:2606.22606 (2026).
<a id="reference-34"></a>
34. Huang, H., et al. (2024). [Fine-tuned Judge Model Is Not a General Substitute for GPT-4](https://arxiv.org/abs/2403.02839). arXiv:2403.02839.
<a id="reference-35"></a>
35. Asai, A., et al. (2024). [OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs](https://arxiv.org/abs/2411.14199). arXiv:2411.14199; Nature (2026).
<a id="reference-36"></a>
36. Jin, B., et al. (2025). [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). arXiv:2503.09516.
<a id="reference-37"></a>
37. Liu, et al. (2026). [Search, Do Not Guess: Teaching Small Language Models to Be Search Agents](https://arxiv.org/abs/2604.04651). arXiv:2604.04651.
<a id="reference-38"></a>
38. Shaikh (2026). [Dissecting Agentic RAG: Component Ablation with a Local 7B Model](https://arxiv.org/abs/2606.21553). arXiv:2606.21553.
<a id="reference-39"></a>
39. Chen, L., Zaharia, M., & Zou, J. (2023). [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176). arXiv:2305.05176.
<a id="reference-40"></a>
40. Ong, I., et al. (2025). [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665). arXiv:2406.18665.
<a id="reference-41"></a>
41. Ding, D., et al. (2024). [Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing](https://arxiv.org/abs/2404.14618). ICLR 2024; arXiv:2404.14618.
<a id="reference-42"></a>
42. Belcak, P., et al., NVIDIA (2025). [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153). arXiv:2506.02153.
<a id="reference-43"></a>
43. [On-Premise LLM Deployment: A Cost-Benefit Analysis](https://arxiv.org/abs/2509.18101). arXiv:2509.18101 (2025).
<a id="reference-44"></a>
44. Lopez, I., et al. (2025). [CLEAR: Clinical Entity Augmented Retrieval](https://www.nature.com/articles/s41746-024-01377-1). npj Digital Medicine.
<a id="reference-45"></a>
45. [BM25 Wins at Scale](https://arxiv.org/abs/2607.26497). arXiv:2607.26497 (2026).
<a id="reference-46"></a>
46. Shah, J. (undated). [Can Dense Retrieval Beat BM25 for Entity Resolution?](https://jayshah.dev/posts/entity-resolution-dense-retrieval/) Blog, non-peer-reviewed.
<a id="reference-47"></a>
47. [LC-ICL: Label-Guided Contrastive In-Context Learning](https://arxiv.org/abs/2606.29407). arXiv:2606.29407 (2026).
<a id="reference-48"></a>
48. Wan, Z., et al. (2023). [GPT-RE: In-context Learning for Relation Extraction using Large Language Models](https://arxiv.org/abs/2305.02105). EMNLP 2023; arXiv:2305.02105.
<a id="reference-49"></a>
49. [DLISC: Schema-aware Information Extraction with On-Device LLMs](https://arxiv.org/abs/2505.14992). arXiv:2505.14992 (2025).
<a id="reference-50"></a>
50. [SchemaRAG](https://arxiv.org/abs/2607.00008). arXiv:2607.00008 (2026).
<a id="reference-51"></a>
51. [Grounded Event Extraction from SEC 8-K Filings](https://arxiv.org/abs/2607.08346). arXiv:2607.08346 (2026).
<a id="reference-52"></a>
52. [Schema-Grounded LLM Extraction for FHIR Digital Twins](https://arxiv.org/abs/2601.05847). arXiv:2601.05847 (2026).
<a id="reference-53"></a>
53. Liu, F., et al. (2021). [Self-Alignment Pretraining for Biomedical Entity Representations (SapBERT)](https://arxiv.org/abs/2010.11784). NAACL 2021; arXiv:2010.11784.
<a id="reference-54"></a>
54. Peeters, R., Steiner, A., & Bizer, C. (2024). [Entity Matching using Large Language Models](https://arxiv.org/abs/2310.11244). arXiv:2310.11244.
<a id="reference-55"></a>
55. [AnyMatch: Efficient Zero-Shot Entity Matching with a Small Language Model](https://arxiv.org/abs/2409.04073). arXiv:2409.04073 (2024).
<a id="reference-56"></a>
56. [Cost-Efficient RAG for Entity Matching (CE-RAG4EM)](https://arxiv.org/abs/2602.05708). arXiv:2602.05708 (2026).
<a id="reference-57"></a>
57. Wang, T., et al. (2025). [Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching (ComEM)](https://arxiv.org/abs/2405.16884). COLING 2025; arXiv:2405.16884.
<a id="reference-58"></a>
58. [Deep Retrieval at CheckThat! 2025](https://arxiv.org/abs/2505.23250). arXiv:2505.23250 (2025).
<a id="reference-59"></a>
59. [Provence: Efficient and Robust Context Pruning for Retrieval-Augmented Generation](https://arxiv.org/abs/2501.16214). arXiv:2501.16214 (2025).
<a id="reference-60"></a>
60. Xu, F., Shi, W., & Choi, E. (2023). [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408). arXiv:2310.04408.
<a id="reference-61"></a>
61. Yue, X., et al. (2024). [AttributionBench: How Hard is Automatic Attribution Evaluation?](https://arxiv.org/abs/2402.15089). arXiv:2402.15089.
<a id="reference-62"></a>
62. [Literature-Grounded Novelty Assessment (Idea Novelty Checker)](https://arxiv.org/abs/2506.22026). SDP 2025; arXiv:2506.22026.
<a id="reference-63"></a>
63. [ReviewGrounder](https://arxiv.org/abs/2604.14261). arXiv:2604.14261 (2026).
<a id="reference-64"></a>
64. Liu, Z., et al. (2026). [Can LLMs Design High-Quality Experiments? (SCOPE/OptED)](https://arxiv.org/abs/2608.03501). arXiv:2608.03501.
<a id="reference-65"></a>
65. Li, et al. (2025). [AgentExpt](https://arxiv.org/abs/2511.04921). arXiv:2511.04921.
<a id="reference-66"></a>
66. [Evaluating RAG Variants for SQL and API Call Generation](https://arxiv.org/abs/2602.07086). arXiv:2602.07086 (2026).
<a id="reference-67"></a>
67. Gao, T., Yen, H., Yu, J., & Chen, D. (2023). [Enabling Large Language Models to Generate Text with Citations (ALCE)](https://arxiv.org/abs/2305.14627). EMNLP 2023; arXiv:2305.14627.
<a id="reference-68"></a>
68. Bohnet, B., et al. (2023). [Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models (AutoAIS)](https://arxiv.org/abs/2212.08037). arXiv:2212.08037.
<a id="reference-69"></a>
69. [Retrieval-Augmented Agentic Rubric Generation for Medical Response Evaluation](https://arxiv.org/abs/2601.15161). arXiv:2601.15161 (2026).
<a id="reference-70"></a>
70. [When Retrieval Helps and Distracts: Biomedical Claim Verification](https://arxiv.org/abs/2608.01409). arXiv:2608.01409 (2026).
<a id="reference-71"></a>
71. [Time to REFLECT: Can We Trust LLM Judges for Evidence-based Research Agents?](https://arxiv.org/abs/2605.19196). arXiv:2605.19196 (2026).
<a id="reference-72"></a>
72. [Retrieval Augmentation for Lay Language Generation (RALL/CELLS)](https://arxiv.org/abs/2211.03818). arXiv:2211.03818.
<a id="reference-73"></a>
73. You, et al. (2025). [PlainQAFact](https://arxiv.org/abs/2503.08890). arXiv:2503.08890.
<a id="reference-74"></a>
74. [Trie Automata for Constrained Decoding over Large Finite Sets](https://arxiv.org/abs/2608.12574). arXiv:2608.12574 (2026).
<a id="reference-75"></a>
75. [Efficient Terminology Integration for LLM-based Translation](https://aclanthology.org/2024.wmt-1.51/). WMT 2024.
<a id="reference-76"></a>
76. Jaswal (2025). [It Takes Two: Terminology-Aware Translation](https://arxiv.org/abs/2511.07461). arXiv:2511.07461.
<a id="reference-77"></a>
77. [Zero-Shot Cross-Lingual Biomedical Concept Normalization](https://www.medrxiv.org/content/10.1101/2025.02.27.25323007v1.full). medRxiv (2025).
<a id="reference-78"></a>
78. [CompactQE: Interpretable Translation Quality Estimation via Small Open-Weight LLMs](https://arxiv.org/abs/2605.15763). arXiv:2605.15763 (2026).
<a id="reference-79"></a>
79. [Copy-Paste to Mitigate Large Language Model Hallucinations](https://arxiv.org/abs/2510.00508). arXiv:2510.00508.
<a id="reference-80"></a>
80. [Code-Switching Information Retrieval](https://arxiv.org/abs/2604.17632). arXiv:2604.17632 (2026).
<a id="reference-81"></a>
81. [The Multilingual Curse at the Retrieval Layer: Evidence from Amharic](https://arxiv.org/abs/2605.24556). arXiv:2605.24556 (2026).
<a id="reference-82"></a>
82. Ma, Z., Kim, D., & Chen, T.-H. (2024). [LibreLog](https://arxiv.org/abs/2408.01585). arXiv:2408.01585.
<a id="reference-83"></a>
83. Xiao, Y., Le, V.-H., & Zhang, H. (2024). [LogBatcher](https://arxiv.org/abs/2406.06156). arXiv:2406.06156.
<a id="reference-84"></a>
84. [Architecture Matters More Than Scale: Financial QA under SME Compute Constraints](https://arxiv.org/abs/2604.17979). arXiv:2604.17979 (2026).
<a id="reference-85"></a>
85. [LLMStructBench](https://arxiv.org/abs/2602.14743). arXiv:2602.14743 (2026).
