# Which Agents Go Local?

**Thesis.** The pipeline's role contracts and deterministic commit boundary make backend substitution safe to *try* for most roles and safe to *deploy* for none until each role is measured against an explicit acceptable error rate. Evidence from related tasks marks Extraction, the Elaborator and the Reader Translator as the first candidates, the Evidence Reviewer and Experiment Validator as conditional candidates, the Critic and Experiment Designer as conditional integrations, and the Synthesist as the role to keep on a frontier model until a specialised local alternative has been compared directly.

**Scope and conventions.** "Local" describes where inference runs and "open-weight" describes weight availability; neither establishes capability, licensing freedom or serving cost, and comparison with a historical cloud checkpoint is not parity with the current frontier. The candidate range is 4–32B parameters, with smaller specialist encoders considered separately. The method combined inspection of role contracts with a targeted narrative literature review; no reproducible search protocol or formal quality appraisal was recorded, and no reviewed study evaluates the complete pipeline or these composite roles end to end. **Near analogue** marks evidence from a related task, domain or model configuration; **proposal** marks an integration or policy that needs direct evaluation. **Candidate** means suitable for evaluation, **conditional candidate** adds required controls, and **retain pending evaluation** is a provisional policy, not proof that local deployment is impossible. The [role configuration](../../src/config.py) already selects compatible backends, including inherited defaults; specialist encoders, proposed roles and new validation controls need engineering. Hardware, context length, concurrency and inference budget must be fixed before any feasibility judgment.

## 1. What the architecture guarantees

The [graph commit validator](../../src/validator.py) enforces five checks: schema validity, resolvable references, a current base-state hash, an allowed status transition, and an unused idempotency key. Schema-constrained decoding satisfies the first and none of the others, and none of the five establishes scientific truth. The boundary mitigates some state-management failures; it does not remove the reasoning, communication and verification failures catalogued in MAST <sup>[10](#reference-10)</sup>.

Agents grade and rules decide. The Evidence Reviewer supplies grades to deterministic verdict fusion with a 0.5 floor and a 0.025 near-tie margin; Critic grades feed median aggregation and Borda rank fusion. Those rules need both useful discrimination between cases and appropriate calibration, and mapping discrete rubric anchors to floats does not produce calibrated probabilities. The Experiment Validator is a model-backed plan reviewer, distinct from the commit validator; keeping it on a frontier model while testing a local Designer is a conservative policy, and different vendors or capability classes do not by themselves guarantee independent errors. Two wiring facts matter for any substitution experiment: changing the `builder` backend also changes the first Critic seat and Extraction unless their wiring is separated, and without a dedicated `critic_panel` block the second and third seats inherit the same backend. Record every effective backend.

## 2. Per-role proposals

![Seven agent roles plotted by creativity and grounding, with bubble size and colour identifying personal computer, GPU workstation, or hosted frontier service.](figures/agent-model-capacity-map.svg)

*Positions are qualitative contributions to generated hypotheses; bubble size and colour identify deployment categories, not parameter counts. Critic and Validator placements cover their local candidates.*

| Role | Proposed status | Local candidate to evaluate | Main condition and risk |
|---|---|---|---|
| Extraction | Candidate | Task-tuned 7–9B, or a specialist encoder | Compare specialist and generative extractors on exact spans, relation direction and domain transfer |
| Evidence Reviewer | Conditional candidate | 14–32B reasoning-tuned; specialist NLI benchmarked separately | Evaluate entailment and study-design assessment separately; measure downstream verdict errors on adjudicated edges |
| Synthesist | Retain pending evaluation | Specialised 4–32B alternatives for comparison | Compare on hypothesis quality, evidence coverage and quote fidelity with equivalent evidence access |
| Critic | Conditional candidate | One 32B reasoning-tuned seat beside a strong reviewer | Measure complementary errors first; audit agreed decisions as well as escalations |
| Elaborator | Candidate | About 7B | Test semantic preservation separately from fluency and readability |
| Reader Translator | Candidate | 7–13B translation models on supported language pairs | Evaluate each language pair, technical vocabulary and passage length |
| Translation Verifier | Conditional candidate | Quality-estimation models | Measure segment-level omissions and meaning changes; generation scores do not validate a verifier |
| Experiment Designer | Conditional candidate | 32B start; 14B lower-cost candidate; 7B exploratory baseline | Use structured grounding; initially keep the frontier Experiment Validator |
| Experiment Validator | Conditional candidate | 8–32B judge-tuned | Test false acceptance and rejection against independent judgments; add final-plan review first |
| Profile Compiler (proposed) | Conditional candidate | Fine-tuned extractor | Test multilingual and compositional inputs; user confirmation is an additional check |
| Result Interpreter (proposed) | Conditional candidate | Code-specialised local model behind numeric gates | Validate source coordinates, outcome labels, units, arms and time points; recompute derived values |

The Synthesist and Critic need the strongest models because they contribute most to creativity and grounding; Extraction supplies the seed structure, the experiment agents contribute after selection, and the Elaborator communicates existing content. Two implementation facts shape the grounding picture: the current workflow applies Evidence review to seed edges before generation, not to newly generated hypotheses, and the [Experiment Validator loop](../../src/cycles/experiment.py) can return the last revision without another review, so final-plan assessment and explicit acceptance criteria must be added before the loop counts as a safeguard.

**Deployment categories** are planning targets, not proven minima. They assume dense models, 4-bit weights, one resident model and low concurrency; context length and runtime overhead add memory.

| Category | Planning estimate | Roles |
|---|---|---|
| Personal computer | CPU with 16–32 GB RAM; GPU optional for the 7–9B candidates | Extraction, Elaborator |
| GPU workstation | One GPU with 24–48 GB VRAM, or roughly 32–64 GB unified memory | Evidence Reviewer, the Critic's local seat, Experiment Designer, Experiment Validator |
| Hosted frontier service | Provider-managed, accessed through an API | Synthesist, provisionally |

The estimates follow llama.cpp's CPU and GPU support <sup>[32](#reference-32)</sup> and Qwen's measured memory: a 32B AWQ model used about 18.7 GB at the shortest tested input and 31.6 GB at roughly 30k input tokens <sup>[33](#reference-33)</sup>.

## 3. The evidence behind the proposals

All rows are near analogues. Each result motivates a direct test; none validates a deployment.

| Role | Study and result | What it does not show |
|---|---|---|
| Extraction | GLiNER motivates specialist extraction <sup>[1](#reference-1)</sup>. In PubMedCausal, PubMedBERT scores 0.7391 F1 on detecting causal language while DeepSeek-R1-32B scores 0.2383 exact pair F1 on extracting cause–effect pairs <sup>[2](#reference-2)</sup> | These are different tasks and metrics, not a shared ceiling, and neither covers the whole graph contract |
| Evidence Reviewer | Košprdić et al. support testing fine-tuned NLI models for scientific claim verification <sup>[3](#reference-3)</sup> | Entailment, risk of bias and certainty need separate evaluation; human–human agreement elsewhere is not a model ceiling; a large model's failure does not make a small one adequate |
| Synthesist | NoLiMa: Llama 3.1 8B drops from 76.7% to 14.2% at 32K tokens on latent-association retrieval <sup>[4](#reference-4)</sup>; OpenScholar synthesises literature well with a specialised 8B retrieval system <sup>[5](#reference-5)</sup> | Neither result bounds the 4–32B range on this multi-turn task |
| Elaborator, Translator, Verifier | ReLay: GPT-4o's factuality fell from 0.671 to 0.515 under metadata-prompted personalisation <sup>[6](#reference-6)</sup>; xCOMET localises translation errors <sup>[7](#reference-7)</sup> | Fluent output can still alter negation, uncertainty, numbers or causal strength; evaluate per language pair |
| Experiment Designer | AutoSDT: 7B fine-tuning lowered ScienceAgentBench success from 3.3% to 2.3% while raising valid execution from 19.9% to 27.5% and DiscoveryBench matching from 4.8% to 6.3% <sup>[8](#reference-8)</sup> | Mixed evidence from discovery coding, not a 14B floor for planning; compare 7B, 14B and 32B directly |
| Result Interpreter | Yun et al. document substantial errors in numeric extraction from trial reports <sup>[9](#reference-9)</sup> | A number's presence somewhere in a source does not show that it belongs to the right result |

**Format, capability and serving.** Gemma 3's IFEval scores are nearly flat from 4B to 27B <sup>[11](#reference-11)</sup>, which describes one family on one benchmark and says nothing about saturation in general. Separate reasoning from formatting: the Format Tax study motivates free-form reasoning followed by serialisation <sup>[12](#reference-12)</sup>, and in the Constraint Tax's sub-3B aggregate a hard answer-only schema raised schema validity from 61.5% to 100% while answer accuracy fell from 19.7% to 11.0% and wrong-but-valid outputs rose from 49.5% to 88.9%, partly because malformed answers became valid <sup>[13](#reference-13)</sup>. Structured output also reduces answer diversity on tasks with several valid answers, which is a reason to test JSON's effect on the Critic, not a demonstrated grading effect <sup>[14](#reference-14)</sup>. Weight-only 4-bit quantisation is a candidate configuration, not the only permissible one <sup>[15](#reference-15)</sup>, and RULER's effective context is a benchmark-defined threshold rather than a universal fraction of advertised length <sup>[16](#reference-16)</sup>. Efficiency claims need a workload definition: model and quantisation versions, hardware, weights plus KV cache and runtime memory, input and reasoning tokens, concurrency, retries, latency, and cost per accepted result; a training-run bill that excludes labelling, teacher calls and evaluation is not the cost. JSONSchemaBench separates schema coverage from compliance on supported schemas, so report both with engine versions and this pipeline's schemas <sup>[17](#reference-17)</sup>.

## 4. Three silent failure modes

A weaker backend rarely breaks this system openly. It shifts distributions that fixed rules then act on.

| Failure mode | What goes wrong | Measurement or control |
|---|---|---|
| Score drift | A backend changes score distributions, so fixed-threshold verdicts move even when rankings do not | Compare discrimination, calibration, false acceptance and false rejection on held-out adjudicated cases; changing POPPER's backbone changed its Type-I error in the same way <sup>[18](#reference-18)</sup> |
| Shared judge errors | Judges agree wrongly, so disagreement-based escalation never fires; the current escalation function triggers only on first-ranked, already_established and not_judgeable_by_field disagreements | Measure complementary errors and audit a sample of agreements; Kim et al. report about 60% agreement conditional on both models being wrong, and vendor or size differences do not ensure independence <sup>[19](#reference-19)</sup> |
| Valid structure, incorrect meaning | A well-formed claim, quote or number is misattributed or misinterpreted | Check exact source spans and structured provenance; for numbers, validate the outcome, unit, population, arm and time point and recompute derived values |

Replay reveals changes on logged cases. It does not say which model is right, and a finite replay set cannot guarantee detection of future distribution shift.

## 5. Retrieval for local agents

The full analysis, including what retrieval can and cannot substitute for, is in [Retrieval for Local Agents](retrieval-for-local-agents-decision.md).

Here **retrieval** selects records from a corpus, **lookup** addresses an exact or keyed store, **verification** scores support, and **orchestration** decides when and how they run. The anchor result is Sen et al.'s "Is Grep All You Need?": on a 116-question LongMemEval subset, inline lexical retrieval beat inline vector retrieval for every tested harness–model pair, and delivery mode moved accuracy as much as the retriever did <sup>[20](#reference-20)</sup>. The models were proprietary and the tasks literal-span, so transfer to local 4–32B readers is a hypothesis, most plausible where the evidence is a literal span.

| Role | Retrieval component to test | Scoped evidence |
|---|---|---|
| Extraction | Retrieved demonstrations and source-span checks | LC-ICL: Llama-3.1-8B relation-extraction F1 rose from 15.7 to 38.7 and from 8.3 to 21.5 with retrieved demonstrations <sup>[21](#reference-21)</sup>; absolute quality and domain transfer still need evaluation |
| Evidence Reviewer | Hybrid lexical and dense retrieval, reranking, specialist verification | An NLI-plus-lexical evaluator matched GPT-4o on reference-based QA evaluation <sup>[22](#reference-22)</sup>; that is not long-context scientific entailment |
| Critic | Exact support lookup, then entailment assessment | Quote existence and semantic support are separate checks; no direct small-model result exists for cross-paper novelty comparison |
| Experiment Designer | Curated structured grounding with bounded search | SCOPE/OptED: unrestricted search gave no significant gain for five of seven models and raised one model's critical-error rate <sup>[23](#reference-23)</sup> |
| Experiment Validator | Retrieve supporting passages, then verify claims | FActScore's retrieval cut aggregate estimation error, not per-claim error <sup>[24](#reference-24)</sup> |
| Reader Translator, Profile Compiler | Glossary, alias and schema lookup | The combined pipeline is untested; lookup does not guarantee correct composition |
| Result Interpreter | Structured lookup, template caches, provenance checks | Log-parsing analogues motivate scaffolding; none validates scientific interpretation |
| Translation Verifier | None identified | Absence of evidence, not evidence of no benefit |

Design rules to test, distilled from the retrieval literature:

- **Route by evidence type.** Literal-span work (quote existence, aliases, glossary terms, numeric cells) goes to inline lexical or exact-match lookup; paraphrase work (entailment, methods grounding) goes to hybrid retrieval with a small reranker; finite sets (glossaries, schemas) go to deterministic dictionary lookup.
- **Deliver evidence inline to untrained small models.** Use multi-step file-based or model-driven search loops only after direct validation or task-specific training. Shaikh's 7B HotpotQA ablation reached almost the same exact match with two fixed steps as with five, and its fixed hybrid retriever beat its heuristic selector, which supports a bounded schedule as the baseline rather than a ban on model-driven search <sup>[26](#reference-26)</sup>.
- **Offload narrow judgments to small specialists** (NLI classifiers, rerankers, span locators, entity linkers) in a cascade, without assuming that every small classifier beats a larger generative model.
- **Shortlist, then adjudicate.** For matching and deduplication, retrieve a small candidate set before any model adjudication.
- **Verify spans and numbers deterministically, and make abstention a schema branch.** Require an explicit "nothing retrieved" field with evidence-conflict checks. Misleading retrieved context can impair abstention, and AbstentionBench studies unanswerable questions rather than retrieval, so test missing, irrelevant, misleading and conflicting evidence separately <sup>[27](#reference-27)</sup>.
- **Curate and cap passages, place evidence for the current claim first, and validate compression per reader.** One near-miss distractor can cost more than a retriever swap gains; Lost in the Middle motivates testing placement <sup>[28](#reference-28)</sup>, and compression gains vary by reader and dataset with no established breakpoint <sup>[29](#reference-29)</sup>.
- **Benchmark 7B, 14B and 32B directly for retrieval-heavy grading.** Pandey finds severe failures to use even gold evidence in untuned models up to 7B and does not test a 14B transition <sup>[25](#reference-25)</sup>.
- **Give the deterministic core final acceptance authority.** Validators, caches, string checks and fixed schedules supplied the reliability controls across the reviewed systems.

## 6. Evaluation roadmap

1. **Baseline.** Define per-role correctness criteria, acceptable error rates and the hardware and workload envelope. Separate training, calibration and held-out cases, and adjudicate independently rather than treating frontier output as ground truth.
2. **Pilot extraction and presentation roles.** Compare specialist and generative extractors; assess rewrites and translations for semantic preservation; benchmark schema support, constrained decoding and reasoning-then-serialisation on each candidate. No literature-established cutoff requires a particular interface at 8B.
3. **Evaluate the Evidence Reviewer, Experiment Validator and Translation Verifier.** Replay adjudicated cases, inspect score distributions, measure downstream errors, and select any threshold or score-mapping change on calibration data before evaluating it on held-out cases.
4. **Test the conditional integrations.** Measure Critic error overlap before adding a local seat; test a grounded local Designer with the retained frontier Validator; build provenance-aware numeric gates for the Result Interpreter and confirmation checks for the Profile Compiler.
5. **Revisit the Synthesist.** Compare specialised local systems with the current backend under equivalent evidence access and recorded inference budgets, including passage re-injection or chunking across turns, quote fidelity and hypothesis quality.

The measurement harness is buildable from existing assets: an Evidence replay against independent judgments; a verbatim-fidelity curve that character-diffs quoted spans against in-context sources across backends and context lengths; a Critic error-overlap audit; role-swap and interaction ablations with recorded inference budgets; a schema-content audit of wrong-but-valid outputs; and a final-plan acceptance evaluation. In the worked cost example, the Evidence Reviewer makes about two-thirds of core calls and the Synthesist holds the deepest context, so those two roles bound the savings. Engineering preconditions for any deployment are to decouple reasoning from serialisation, grade in discrete anchors mapped deterministically to floats, evaluate the fusion thresholds on labelled calibration data, benchmark the actual delta schemas against the candidate grammar engines, add content gates for quotes and numbers, and add review of the final experiment plan with an explicit acceptance policy.

**First two measurements:** the Evidence replay on independently adjudicated edges and the Critic error-overlap audit. Later deployment decisions follow their results, with ongoing sampling to catch failures outside the replay set.

**Distillation and licensing.** Fine-tuning on the pipeline's prompt-hashed logs is attractive, and the boundary is contractual. Check the teacher's terms and the intended student use: Anthropic's output-training policy permits certain non-competing specialised uses, including information extraction, while restricting competing-model development <sup>[30](#reference-30)</sup>, and the Llama 3.1 Community License permits output-based training subject to conditions, including naming requirements for some distributed models <sup>[31](#reference-31)</sup>. Check each later release separately. Policy links were checked on 12 September 2026.

## References
<a id="reference-1"></a>
1. [GLiNER](https://arxiv.org/abs/2311.08526). arXiv:2311.08526.
<a id="reference-2"></a>
2. Kunle-John et al. [PubMedCausal](https://arxiv.org/html/2605.28363v1), Tables 4–5. arXiv:2605.28363.
<a id="reference-3"></a>
3. Košprdić et al. (2024). [Scientific Claim Verification with Fine-Tuned NLI Models](https://www.scitepress.org/Papers/2024/129000/129000.pdf).
<a id="reference-4"></a>
4. Modarressi et al. [NoLiMa](https://arxiv.org/html/2502.05167v1), Table 3. arXiv:2502.05167.
<a id="reference-5"></a>
5. [OpenScholar](https://arxiv.org/html/2411.14199v1). arXiv:2411.14199.
<a id="reference-6"></a>
6. [ReLay](https://arxiv.org/html/2605.00468v1), Table 2. arXiv:2605.00468.
<a id="reference-7"></a>
7. [xCOMET](https://arxiv.org/abs/2310.10482). arXiv:2310.10482.
<a id="reference-8"></a>
8. Li et al. [AutoSDT](https://arxiv.org/html/2506.08140v1), Table 3. arXiv:2506.08140.
<a id="reference-9"></a>
9. Yun et al. [Numeric extraction from randomised controlled trials](https://arxiv.org/abs/2405.01686). arXiv:2405.01686.
<a id="reference-10"></a>
10. Cemri et al. [Why Do Multi-Agent LLM Systems Fail? (MAST)](https://arxiv.org/abs/2503.13657). arXiv:2503.13657.
<a id="reference-11"></a>
11. [Gemma 3 technical report](https://arxiv.org/html/2503.19786v1), Table 18. arXiv:2503.19786.
<a id="reference-12"></a>
12. Lee, D'Antoni, & Berg-Kirkpatrick. [The Format Tax](https://arxiv.org/abs/2604.03616v1). arXiv:2604.03616.
<a id="reference-13"></a>
13. [The Constraint Tax](https://arxiv.org/html/2605.26128v1), Table 3. arXiv:2605.26128.
<a id="reference-14"></a>
14. [Structured Output Collapses Answer Diversity Across 44 Language Models](https://arxiv.org/html/2607.18476v1). arXiv:2607.18476.
<a id="reference-15"></a>
15. [Quantization Hurts Reasoning?](https://arxiv.org/html/2504.04823v1). arXiv:2504.04823.
<a id="reference-16"></a>
16. [RULER](https://arxiv.org/html/2404.06654v1). arXiv:2404.06654.
<a id="reference-17"></a>
17. [JSONSchemaBench](https://arxiv.org/html/2501.10868v1). arXiv:2501.10868.
<a id="reference-18"></a>
18. Huang et al. [POPPER](https://arxiv.org/pdf/2502.09858v1), Table 4. arXiv:2502.09858.
<a id="reference-19"></a>
19. Kim et al. [Correlated Errors in Large Language Models](https://arxiv.org/html/2506.07962v1). arXiv:2506.07962.
<a id="reference-20"></a>
20. Sen, Kasturi, Lumer, Gulati, & Subbiah (2026). [Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](https://arxiv.org/html/2605.15184v1), Table 1. arXiv:2605.15184.
<a id="reference-21"></a>
21. [LC-ICL](https://arxiv.org/abs/2606.29407). arXiv:2606.29407.
<a id="reference-22"></a>
22. Balamurali & Cheng. [Revisiting NLI for long-form QA evaluation](https://arxiv.org/html/2511.07659v1). arXiv:2511.07659.
<a id="reference-23"></a>
23. Liu et al. [SCOPE/OptED](https://arxiv.org/html/2608.03501v1). arXiv:2608.03501.
<a id="reference-24"></a>
24. Min et al. (2023). [FActScore](https://arxiv.org/html/2305.14251v1), Table 6. EMNLP 2023.
<a id="reference-25"></a>
25. Pandey. [Retrieval-utilization study](https://arxiv.org/html/2603.11513v1). arXiv:2603.11513.
<a id="reference-26"></a>
26. Shaikh. [Local 7B HotpotQA ablation](https://arxiv.org/html/2606.21553v1). arXiv:2606.21553.
<a id="reference-27"></a>
27. [AbstentionBench](https://arxiv.org/html/2506.09038v1). arXiv:2506.09038.
<a id="reference-28"></a>
28. Liu et al. [Lost in the Middle](https://arxiv.org/abs/2307.03172). arXiv:2307.03172.
<a id="reference-29"></a>
29. [Fixed RAG Compression Collapses Measured Reader Scaling](https://arxiv.org/html/2606.21807v1). arXiv:2606.21807.
<a id="reference-30"></a>
30. Anthropic. [Can I use my outputs to train an AI model?](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model) Help Center, checked 12 September 2026.
<a id="reference-31"></a>
31. Meta. [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
<a id="reference-32"></a>
32. [llama.cpp](https://github.com/ggml-org/llama.cpp). CPU and GPU support.
<a id="reference-33"></a>
33. [Qwen 2.5 speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html). Measured memory examples.
