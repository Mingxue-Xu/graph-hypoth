# Retrieval for Local Agents

*Decision version. The complete report is [retrieval-for-local-agents.md](retrieval-for-local-agents.md). Companion to [Which Agents Go Local?](which-agents-go-local.md).*

**Thesis.** Retrieval substitutes for one thing a small model lacks, stored knowledge, and does that well. It does not substitute for reading, for resolving conflicts between evidence and prior belief, for judgement over paraphrased text, for synthesis, or for deciding when to search. Deterministic code substitutes for a second class of work, literal verification and lookup. What is left for a model in every role is reading natural-language evidence into a typed judgement, and whether a small model suffices for that is a question about the reader, not the retriever.

**Conventions.** Evidence is labelled direct, near analogue, mechanistic inference or proposal; almost everything here is a near analogue, and no study evaluates these composite roles end to end. Retrieval selects from a corpus; lookup addresses a keyed store; verification scores support; orchestration schedules and delivers.

## 1. What retrieval can and cannot substitute for

**It substitutes for stored knowledge.** RETRO at 7.5B matched GPT-3 with about 25 times fewer parameters <sup>[2](#reference-2)</sup>; Atlas at 11B beat a 540B closed-book model on NaturalQuestions with 64 examples <sup>[3](#reference-3)</sup>; long-tail facts are learned poorly at any scale and retrieval reduces the dependence <sup>[4](#reference-4), [5](#reference-5)</sup>; a trained model stores about two bits of knowledge per parameter, so capacity for facts is roughly linear in size <sup>[6](#reference-6)</sup>. This is the sense in which retrieval "fulfils a lot", and it matters least here, because the pipeline's prompts already minimise dependence on stored knowledge.

**It does not substitute for reading.** Untuned models at 7B and below fail to use even oracle context, and no study tests a 14B transition <sup>[7](#reference-7)</sup>; reader size still moves retrieval-grounded scores <sup>[8](#reference-8)</sup>; position, distraction and near-miss distractors cost 15–30 points and up to 25 points respectively <sup>[9](#reference-9), [10](#reference-10), [11](#reference-11)</sup>; training the reader on distractors helps, which is a change to the model <sup>[14](#reference-14)</sup>; and delivery mode moved accuracy as much as the retriever did <sup>[16](#reference-16)</sup>.

**It does not resolve conflicts between evidence and memory.** Models resolve context–memory conflicts inconsistently <sup>[17](#reference-17)</sup>, and abstention prompts still elicited answers on 41.6% of misleading-context cases in 3.8B–8B models <sup>[21](#reference-21)</sup>. Deterministic verdict fusion contains this only partly; calibration is a precondition for any smaller reader.

**Deterministic code substitutes for literal judgement.** Span existence, citation-ID resolution, numeric recomputation and schema checks remove decisions from the model entirely; mechanical citation checking reached 92% accuracy with zero hallucinated citations where retrieval-only systems fabricated 17% <sup>[27](#reference-27)</sup>. This is what makes small models safe where a literal check catches the dominant error.

**What remains is reading into a typed judgement,** and a small model can be enough for narrow forms of it: an NLI evaluator matched GPT-4o on QA evaluation (83.57% versus 83.60%) <sup>[28](#reference-28)</sup>, small extractors beat frontier zero-shot after task-specific training <sup>[31](#reference-31), [32](#reference-32), [33](#reference-33)</sup>, an 8B model with a retrieval datastore matched GPT-4o on scientific literature synthesis <sup>[35](#reference-35)</sup>, and 3B–7B models can be trained to run search loops <sup>[36](#reference-36)</sup>. Fine-tuned judges generalise poorly beyond their training tasks <sup>[34](#reference-34)</sup>, so a specialist covers one judgement, not a composite role, and novel hypothesis synthesis has no small-model result. Small rather than frontier is a cost, latency and privacy argument backed by routing economics <sup>[39](#reference-39), [40](#reference-40), [42](#reference-42)</sup>, and it holds only where the reader can read.

**Division of labour by role.**

| Role | Retrieval supplies | Deterministic code supplies | Model must supply |
|---|---|---|---|
| Extraction | Demonstrations, schema slices, aliases | Span checks, schema validity, dedup shortlist | Seed text → typed nodes and edges |
| Evidence Reviewer | Hybrid retrieval, reranking, budget | Quote existence, role requirements | Entailment and design grading over paraphrase |
| Critic | Prior work for novelty | Quote existence | Mechanism audit; novelty synthesis (frontier provisionally) |
| Experiment Designer | Curated methods index | Citation-ID enforcement, abstention field | Mechanism → design |
| Experiment Validator | Supporting passages | Lexical pre-filter | Claim–passage support scoring |
| Elaborator, Translator | Definitions, glossary | Constrained decoding, sentence routing | Meaning-preserving rewriting |
| Profile Compiler | Terminology, lexical retrieval for code-switching | Span grounding | Multilingual extraction (fine-tuned only) |
| Result Interpreter | Template caches, structured lookup | Coordinates, numeric gates, recomputation | Table or log → typed evidence |

## 2. Anchor result and per-role answers

On a 116-question LongMemEval subset with proprietary models, inline lexical retrieval beat inline vector retrieval for every harness–model pair, the largest gap was 23.3 points for Gemini 3.1 Flash-Lite, and file-based delivery collapsed one pairing from 93.1% to 55.2% <sup>[16](#reference-16)</sup>. Route literal-span work to lexical or exact lookup and paraphrase work to hybrid retrieval with reranking; deliver inline to untrained small models.

| Role | Retrieval benefit | Mechanism to test | Scoped result |
|---|---|---|---|
| Extraction | Yes | Retrieved demonstrations; schema slicing; alias lookup; verbatim-span verification | Llama-3.1-8B relation-extraction F1 15.7→38.7 and 8.3→21.5 with retrieved demonstrations <sup>[47](#reference-47)</sup>; unsupported-claim rate 8%→about 0% with verbatim checks <sup>[51](#reference-51)</sup> |
| Evidence Reviewer | Yes, with a ceiling | Hybrid BM25 plus dense, reranking, passage budgeting, NLI offload | Recall@5 0.816 versus 0.587 dense-only on scientific-claim retrieval <sup>[58](#reference-58)</sup>; utilisation binds at 7B and below <sup>[7](#reference-7)</sup> |
| Critic audit | Yes | Exact span lookup, then entailment specialist | Vendor-reported 0.1B verifier above GPT-4-0613 by 0.17–2.64 points <sup>[29](#reference-29)</sup> |
| Critic novelty | Retrieval yes; local synthesis untested | Retrieve and rerank prior work | Reviewed systems synthesise with 70–120B models <sup>[62](#reference-62), [63](#reference-63)</sup> |
| Experiment Designer | Curated yes; unrestricted search no | Entity-linked index, small retriever and reranker, short schedule, enforced citation IDs | Search gave no gain for five of seven models and raised one model's redline rate 7.67%→14.00% <sup>[64](#reference-64)</sup>; two fixed steps ≈ five for a 7B model <sup>[38](#reference-38)</sup> |
| Experiment Validator | Yes | Retrieve, then score support; lexical pre-filter | FActScore's retrieval cut aggregate estimation error 39.6→5.1 points, not per-claim error <sup>[23](#reference-23)</sup> |
| Elaborator | Yes | Definition retrieval; verify only elaborated sentences | ROC-AUC 0.727→0.810 <sup>[73](#reference-73)</sup> |
| Reader Translator | Plausible | Exact glossary lookup plus constrained decoding | Components supported separately; composite untested <sup>[74](#reference-74)</sup> |
| Translation Verifier | Unknown | Pilot only if the segment-level gap matters | Open 27–30B models competitive at system level, weaker at spans <sup>[78](#reference-78)</sup> |
| Profile Compiler | Partial | Span verification; lexical retrieval for code-switching; terminology lookup | BM25 more robust than dense under code-switching <sup>[80](#reference-80)</sup> |
| Result Interpreter | Yes, via lookup | Template caches, structured lookup, numeric gates | 8B parser +13.7% relative at 2.7× speed with cache scaffolding <sup>[82](#reference-82)</sup> |

## 3. Design rules and what would change the answer

Route by evidence type; deliver inline; offload narrow judgements to specialists; shortlist then adjudicate; verify spans and numbers deterministically with abstention as a schema branch; curate and cap passages and validate compression per reader; benchmark 7B, 14B and 32B directly; keep final acceptance authority in the deterministic core; and, for every proposed change, name which column of the table above it addresses, because a retriever upgrade cannot fix a reading failure.

The answer changes if utilisation failures persist at 14B on this pipeline's tasks (route the Evidence Reviewer and Validator to larger or trained readers), if a fine-tuned small reader matches the frontier backend on adjudicated edges (retrieval plus a small reader suffices), if a trained 7B search agent beats the fixed two-step schedule (model-driven orchestration becomes viable), or if a specialised 4–32B Synthesist matches the frontier backend with equivalent evidence access. The companion report's harness takes all four measurements.

## References
<a id="reference-2"></a>
2. Borgeaud, S., et al. (2022). [Improving language models by retrieving from trillions of tokens (RETRO)](https://arxiv.org/abs/2112.04426). ICML 2022.
<a id="reference-3"></a>
3. Izacard, G., et al. (2023). [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299). JMLR.
<a id="reference-4"></a>
4. Kandpal, N., et al. (2023). [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411). ICML 2023.
<a id="reference-5"></a>
5. Mallen, A., et al. (2023). [When Not to Trust Language Models](https://arxiv.org/abs/2212.10511). ACL 2023.
<a id="reference-6"></a>
6. Allen-Zhu, Z., & Li, Y. (2024). [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405). ICLR 2025.
<a id="reference-7"></a>
7. Pandey (2026). [Can Small Language Models Use What They Retrieve?](https://arxiv.org/abs/2603.11513). arXiv:2603.11513.
<a id="reference-8"></a>
8. Tan, et al. (2025). [PRGB Benchmark](https://arxiv.org/abs/2507.22927). arXiv:2507.22927.
<a id="reference-9"></a>
9. Liu, N. F., et al. (2024). [Lost in the Middle](https://arxiv.org/abs/2307.03172). TACL 2024.
<a id="reference-10"></a>
10. Shi, F., et al. (2023). [Large Language Models Can Be Easily Distracted by Irrelevant Context](https://arxiv.org/abs/2302.00093). ICML 2023.
<a id="reference-11"></a>
11. Amiraz, C., et al. (2025). [The Distracting Effect](https://arxiv.org/abs/2505.06914). ACL 2025.
<a id="reference-14"></a>
14. Zhang, T., et al. (2024). [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131). arXiv:2403.10131.
<a id="reference-16"></a>
16. Sen, S., Kasturi, Lumer, Gulati, & Subbiah (2026). [Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](https://arxiv.org/abs/2605.15184). arXiv:2605.15184.
<a id="reference-17"></a>
17. Xu, R., et al. (2024). [Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319). EMNLP 2024.
<a id="reference-21"></a>
21. [Prompt-Based Abstention Fails Under Misleading Context](https://arxiv.org/abs/2608.22228). arXiv:2608.22228 (2026).
<a id="reference-23"></a>
23. Min, S., et al. (2023). [FActScore](https://arxiv.org/abs/2305.14251). EMNLP 2023.
<a id="reference-27"></a>
27. [Citation-Grounded Code Comprehension](https://arxiv.org/abs/2512.12117). arXiv:2512.12117 (2025).
<a id="reference-28"></a>
28. Balamurali & Cheng (2025). [Revisiting NLI: Cost-Effective Metrics for QA Evaluation](https://arxiv.org/abs/2511.07659). arXiv:2511.07659.
<a id="reference-29"></a>
29. Vectara (2024–2025). [HHEM-2.1-Open](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model). Vendor blog.
<a id="reference-31"></a>
31. Zaratiana, U., et al. (2024). [GLiNER](https://arxiv.org/abs/2311.08526). NAACL 2024.
<a id="reference-32"></a>
32. Sainz, O., et al. (2024). [GoLLIE](https://arxiv.org/abs/2310.03668). ICLR 2024.
<a id="reference-33"></a>
33. [Sub-Billion, Super-Frontier](https://arxiv.org/abs/2606.22606). arXiv:2606.22606 (2026).
<a id="reference-34"></a>
34. Huang, H., et al. (2024). [Fine-tuned Judge Model Is Not a General Substitute for GPT-4](https://arxiv.org/abs/2403.02839). arXiv:2403.02839.
<a id="reference-35"></a>
35. Asai, A., et al. (2024). [OpenScholar](https://arxiv.org/abs/2411.14199). arXiv:2411.14199; Nature (2026).
<a id="reference-36"></a>
36. Jin, B., et al. (2025). [Search-R1](https://arxiv.org/abs/2503.09516). arXiv:2503.09516.
<a id="reference-38"></a>
38. Shaikh (2026). [Dissecting Agentic RAG: Component Ablation with a Local 7B Model](https://arxiv.org/abs/2606.21553). arXiv:2606.21553.
<a id="reference-39"></a>
39. Chen, L., Zaharia, M., & Zou, J. (2023). [FrugalGPT](https://arxiv.org/abs/2305.05176). arXiv:2305.05176.
<a id="reference-40"></a>
40. Ong, I., et al. (2025). [RouteLLM](https://arxiv.org/abs/2406.18665). arXiv:2406.18665.
<a id="reference-42"></a>
42. Belcak, P., et al., NVIDIA (2025). [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153). arXiv:2506.02153.
<a id="reference-47"></a>
47. [LC-ICL: Label-Guided Contrastive In-Context Learning](https://arxiv.org/abs/2606.29407). arXiv:2606.29407 (2026).
<a id="reference-51"></a>
51. [Grounded Event Extraction from SEC 8-K Filings](https://arxiv.org/abs/2607.08346). arXiv:2607.08346 (2026).
<a id="reference-58"></a>
58. [Deep Retrieval at CheckThat! 2025](https://arxiv.org/abs/2505.23250). arXiv:2505.23250.
<a id="reference-62"></a>
62. [Literature-Grounded Novelty Assessment (Idea Novelty Checker)](https://arxiv.org/abs/2506.22026). SDP 2025.
<a id="reference-63"></a>
63. [ReviewGrounder](https://arxiv.org/abs/2604.14261). arXiv:2604.14261 (2026).
<a id="reference-64"></a>
64. Liu, Z., et al. (2026). [Can LLMs Design High-Quality Experiments? (SCOPE/OptED)](https://arxiv.org/abs/2608.03501). arXiv:2608.03501.
<a id="reference-73"></a>
73. You, et al. (2025). [PlainQAFact](https://arxiv.org/abs/2503.08890). arXiv:2503.08890.
<a id="reference-74"></a>
74. [Trie Automata for Constrained Decoding over Large Finite Sets](https://arxiv.org/abs/2608.12574). arXiv:2608.12574 (2026).
<a id="reference-78"></a>
78. [CompactQE](https://arxiv.org/abs/2605.15763). arXiv:2605.15763 (2026).
<a id="reference-80"></a>
80. [Code-Switching Information Retrieval](https://arxiv.org/abs/2604.17632). arXiv:2604.17632 (2026).
<a id="reference-82"></a>
82. Ma, Z., Kim, D., & Chen, T.-H. (2024). [LibreLog](https://arxiv.org/abs/2408.01585). arXiv:2408.01585.
