# digst

A quick alternative approach to pseudonymising Danish municipal council minutes,
and an argument that the premise was wrong.

In February 2026 the Danish Agency for Digital Government published
[9 million words of council minutes](https://sprogteknologi.dk/dataset/tekstdata-fra-pilotprojekt-om-kommunale-byrads-og-miljo-og-teknikudvalgsmoder)
as training data for Danish language models, with direct personal identifiers
pseudonymised.

## This is not a data privacy problem

The minutes are already public. They stay online at the municipalities' own
agenda portals, indexed and crawlable. Anyone with the resources to train a
language model has the resources to scrape and clean the source, and will prefer
to, because they can fit the cleaning to their own pipeline. Pseudonymising the
published copy protects no one. It only degrades the copy.

The real question is model privacy: whether a trained model will emit personal
data from its weights. That is a property of the output, and pseudonymising the
input does not address it. It does not prevent anyone training on the raw source,
and whatever slips through the filter, as some always does, is still memorisable.

## Randomised identities work against the goal

Memorisation is strongest for sequences that are rare and repeated. A real
person's name in a large corpus is the opposite: it is well attested across many
contexts, so the model learns a diffuse, generalised representation of that
entity rather than reciting the sentences it appeared in.

Replacing that name with a fresh random pseudonym per document inverts both
properties. The real entity loses supporting evidence, becoming sparser and
noisier rather than absent, since the same person is named in news, legal texts
and reference works elsewhere in any realistic training mix. Meanwhile each
pseudonym becomes a rare token sequence bound tightly to one highly specific
passage, which is exactly the regime where verbatim memorisation is strongest.
Randomisation converts one diffuse entity into many sharply bound ones.

It also removes the only thing that made the corpus worth collecting.
Cross-document coreference is what allows a model to learn that a named actor
holds a position and votes a certain way. Pseudonyms scoped to a single agenda
item make the same person a different person in the next document, so the model
can only learn surface phrasing, not the association between actors and
decisions. That is the pre-transformer paradigm.

The address masking has the same shape. `[adresse]` occurs 21,431 times across
70% of rows in a 4,118-row sample, fewer than 1% of them in a context suggesting
a private residence. The rest are roads, towns, districts, institutions and
project names. A model trained on this learns that `[adresse]` is an ordinary
high-frequency Danish word, that one `[adresse]` should be demolished while
another is worth preserving, and it will emit the token in generation.

## What would actually target regurgitation

Deduplication, since memorisation scales with how often a sequence repeats.
Differential privacy during training. Output filtering. Machine unlearning for
specific requests. All of these act where the risk is, which is the model, not
the corpus.

The requirement worth specifying is not "pseudonymise the data" but a measurable
bound on what remains, estimated by independent detection rather than asserted.

## What this repo is

`pseudonymize.py` runs agenda items from
[dagsordener.aarhus.dk](https://dagsordener.aarhus.dk) through a small local
instruction-tuned model (Qwen3 30B-A3B via Ollama), extracting person names,
emails, phone numbers and person-related addresses while leaving institutions,
committees, parties, place names and case numbers intact. Replacements are
corpus-wide and gender-consistent, with a second pass so entities found late are
replaced in earlier documents too.

It took an afternoon. That is the point: the extraction is not the hard part, and
it is not where the problem was.

## Files

- `pseudonymize.py` — the pipeline
- `stikprøve.csv` — 4,118 rows sampled from the published dataset (CC0-1.0)
