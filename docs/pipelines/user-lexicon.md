# Reader Lexicon Procedure

How to build a `ReaderLexicon` artifact — see the neutral
[`example.yaml`](../../examples/reader-lexicons/example.yaml) —
the "what this reader already knows" input that drives the reader-translation seam
(`src/reader_translation.py`) via `ResearchProfile.reader_lexicon`. The seam runs
over the elaborated prose, so the lexicon only takes effect on a synthesist run
invoked with `--trace-dir` (prose generation is enabled by default) — e.g. `graph-hypoth-synthesist
--profile <profile>.yaml --export-dir <run-dir> --trace-dir <run-dir>/trace
--elaborate`. With `--no-elaborate` the lexicon is loaded and validated but never
used, and no card is translated.

This is a one-time, session-run procedure (a person or an agent performs it and a
human reviews the result). It is NOT pipeline code: nothing in `src/` fetches or
digests at run time — the pipeline only loads the finished YAML through the
`ReaderLexicon` schema (`src/research_profile.py`).

## 1. Source

Ground truth is text that evidences the reader's working vocabulary, best of all the
reader's OWN writing:

- Preferred: a local paper directory supplied by the reader (e.g.
  `main.tex` + `references.bib`) — read locally, no web fetch.
Record non-sensitive provenance in the artifact's `source` field and header
comment. Do not commit a private local path. If the supplied paper is not authored
by the reader, say so explicitly in the header — the digest then captures the
vocabulary of a text the reader chose, which only the reader can confirm as
familiar.

## 2. Digest rules (Personalized-Jargon rule)

Condition on the reader's own text — following the Personalized Jargon
Identification insight (NAACL 2024): a researcher's familiarity is best predicted
from their own publications.

- Include ONLY terms that actually occur in the source text. Never add terms the
  text does not use, however plausible ("she surely knows X" is exactly the guess
  this rule forbids).
- `home_field`: the field label the text itself works in (from its title, abstract,
  and keywords), not an externally assumed label.
- `familiar_terms`: the technical terms the text uses and (ideally) defines; write
  each `gloss` close to the text's own wording. Prefer load-bearing terms over
  exhaustive listing.
- `analogy_domains`: the domains the text itself draws examples, methods, or
  contrasts from.
- `papers`: the digested paper's own title (+ year/venue), plus any references in
  the bibliography authored by the reader, if identifiable by author match.

## 3. Artifact format

A YAML mapping that loads through `ReaderLexicon`
(`src.research_profile.ReaderLexicon.model_validate`):

```yaml
home_field: <field label from the text>
familiar_terms:
  - {term: <term as used>, gloss: <short gloss in the text's own words>}
analogy_domains: [<domain>, ...]
papers:
  - {title: <title>, year: <int or omit>, venue: <venue or omit>}
source: <exact provenance (path/URLs)>
generated: "<YYYY-MM-DD>"   # QUOTE the date — bare YAML dates parse as date objects, not str
```

Header comment block (required):

1. `# PENDING-HUMAN-REVIEW` as the first line — the marker stays until the reader
   reviews the artifact.
2. The source path/URLs and the generation date.
3. Any provenance caveat (e.g. paper not authored by the reader).
4. The refresh note (below).

The research-profile tests keep the checked-in example loadable through the
schema.

## 4. Human review (required)

Only the reader can attest to what they know. The reader reviews and edits the
artifact — removing terms they do not actually know, adding vocabulary they do —
before its first use in a run. Keep the `# PENDING-HUMAN-REVIEW` marker until then.

## 5. Refresh on demand

Re-run this procedure when the reader's working vocabulary shifts (new papers, new
field). Staleness is degradation-only by design: a term missing from the lexicon
simply gets no translation (the card keeps the original term with a self-contained
definition); nothing breaks.
