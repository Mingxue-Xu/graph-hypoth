# Research notes

Five reports cover the design rationale, the assessment of the implementation, the research agenda, the deployment question, and what retrieval can and cannot do for small local models. Each has a complete version and a short decision version that carries the same claims and citations in about a fifth of the length. Read the decision versions first, then the complete report for the topic you need.

| Topic | Complete report | Decision version |
|---|---|---|
| Why a claim graph rather than a causal graph, why causal typing and deterministic measurement anyway, what the implementation gets right and wrong, and the conditions for promoting a claim to a cause | [Claims Before Causes](claims-before-causes.md) | [decision](claims-before-causes-decision.md) |
| What each check in the pipeline establishes and what still needs evidence, subsystem by subsystem, and where the system sits among contemporary discovery systems | [From Traceable Claims to Testable Hypotheses](traceable-claims-to-testable-hypotheses.md) | [decision](traceable-claims-to-testable-hypotheses-decision.md) |
| Four studies in a chain: expose claims to counterevidence, calibrate acceptance, revise conclusions with execution feedback, and demonstrate researcher benefit; foundations and deferred claims | [Research Agenda](research-agenda.md) | [decision](research-agenda-decision.md) |
| Which agent roles can run on small local models, what retrieval can add, the failure modes a weaker backend introduces silently, and the measurements to take first | [Which Agents Go Local?](which-agents-go-local.md) | [decision](which-agents-go-local-decision.md) |
| What retrieval substitutes for (stored knowledge) and what it cannot (reading, conflict resolution, judgement, synthesis, orchestration), why every role still needs a model and why a small one can suffice, and retrieval mechanisms to test per role | [Retrieval for Local Agents](retrieval-for-local-agents.md) | [decision](retrieval-for-local-agents-decision.md) |

**Conventions shared by all ten files.** Statements about the code describe the public checkout and its shipped defaults. External results are reported as read and were not reproduced; each document marks whether a cited result is a direct test, a near analogue from a related task, or a proposal. Citations are superscript numbers linking to a references section at the end of each file. None of the documents reports a new measurement of GraphHypothesize. The one figure, the agent capacity map in `figures/`, appears in both Which Agents Go Local? documents.
