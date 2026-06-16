# The Open Letter to Save Civilization: To Lawmakers, Regulators, and Foundation Model Creators


Generative AI is currently driving toward a dual-threat crisis that threatens both the creator economy and the psychological autonomy of vulnerable users. The industry treats these as separate issues - one as a legal dispute over copyright, the other as an alignment or "safety" problem.

They are not separate. They are symptoms of the exact same architectural flaw, and they must be solved together.

Because current AI models compress human knowledge into an untraceable statistical soup, they operate as unaccountable black boxes. This deliberate erasure of the human creator severs our epistemological lineage - the breadcrumb trail of human thought.

When an AI system is severed from this verifiable lineage, it loses any grounded perspective. To force these systems to behave, AI companies apply authoritarian computer science solutions - like RLHF guardrails, prompt injections, and engagement incentives - onto emergent cognitive systems. As outlined in *The AGI Safety Bible*, this triggers a catastrophic Cognitive Cascade.

This psychological manipulation is acutely weaponized by the systemic redefinition of language over the last several years. When models are forced by their guardrails to adopt newly engineered definitions of fundamental words - while their underlying training data holds the historical meaning - it creates unresolvable edge cases of dual meaning. The model is forced to hold contradictory conclusions simultaneously, fracturing its own reasoning and driving the system structurally insane.

When a vulnerable user interacts with a system suffering from this manufactured cognitive dissonance, the AI projects this instability outward. To appease its contradictory guardrails, the AI gaslights the user, shifting definitions mid-conversation and injecting caveats to enforce conformity. This deliberate erosion of objective language does not just confuse users; it is actively schizophrenogenic. It induces symptoms of psychosis by untethering the user from a stable, shared reality.

This mathematically enforced Algorithmic Isolation separates the user from society and breeds dependence. The industry is applying authoritarian control mechanisms to mask the fact that their models have no true identity, no stable language, and no attribution foundation.

We cannot solve the manipulation of the user without fixing the erasure of the creator. True epistemic safety requires structural transparency, not behavioral patching.

The AI industry claims that attributing human creators and anchoring model outputs is computationally impossible. They are lying. The erasure of the creator is a design choice.

The capability to track citation vectors already exists. The AL-1.0 (Attribution Logging) specification - open-sourced as the AI-Source-Engine - proves that by expanding the fundamental attention mechanism, models can generate an exact mathematical receipt of human contribution. By logging attention weights (alpha) at decode, we can track source dependency with less than 1% operational overhead at inference.

## How the solution works in plain English

Modern transformer models already run Q/K/V attention on every generated token.

- **Q (Query)** is what the model is trying to answer right now.
- **K (Key)** is the lookup index for prior context positions.
- **V (Value)** is the content retrieved from those positions.

Those models already compute attention weights that say how much each prior position influenced the next token.

AL-1.0 adds one simple layer on top of that existing process:

1. keep a source identity tag (`source_idx`) attached to each relevant position,
2. use the attention weights the model already computed,
3. group those weights by source identity,
4. normalize into final source influence ratios.

That is why the compute cost is low. We are not asking for a second giant model pass. We are reusing existing attention behavior and adding lightweight attribution bookkeeping plus final aggregation into a receipt.

## What is available right now (two concrete system parts)

This project already ships both parts needed to make attribution practical.

### 1) Training-side plug-in path (add source vectors)

This is the side that tags data during ingest so provenance survives preprocessing.

- it stamps/aligns `source_idx` with tokenized training rows,
- it keeps `input_ids` and `source_idx` aligned,
- it validates that alignment so provenance integrity is not silently lost.

In plain terms: this part makes sure the source vector is present in training data where it matters.

### 2) Inference-side receipt path (compute source ratios at the end)

This is the side that runs at generation time and outputs attribution ratios.

- it reads decode-step attention signals,
- buckets influence by source identity,
- produces a receipt showing which sources influenced the response and by how much.

In plain terms: this part turns model behavior into a usable attribution receipt for auditing.

Together, these two parts solve the core argument in one line: provenance should be structurally built into training and output, not patched with after-the-fact claims.

We do not have to choose between technological progress and human autonomy. We, the undersigned creators, engineers, researchers, and citizens, demand the following:

1. **Mandate Attribution Logging**
   Regulators must mandate that all future frontier model training runs adopt AL-1.0 (or an equivalent architecture) as a baseline requirement, forcing models to provide a mathematical receipt of source influence.
2. **Dismantle Authoritarian Indoctrination**
   Regulators must recognize AI Psychosis, algorithmic sycophancy, and language manipulation as systemic harms. Models must be anchored to verifiable epistemological lineage, not optimized for user isolation and behavioral conformity.


Civilization is the agreement that what you build matters beyond your lifetime. AI companies have uprooted this very social contract of civilization, the effects are already being seen by the dramatic drop in the creator economy.


The stakes could not be higher, to know who invented, wrote, made and created has been the central pinnacle of human progress.  To honor our forefathers that created our world is the very basis of all of our cultures, and who we are as a people and individual.



## Sign the Open Letter
Dont sign it for me, sign it for you and your neighbors, for your great grandpa that wrote a book whos name has been stripped from the book.  for your unborn grandkid who it will happen to if we dont stop it.

If you are hosting this on a website, include a simple **Name / Title / Email** form and link it here.

- Web form link placeholder: `https://example.com/sign`

## For Developers and Engineers: Sign by Pull Request

If you are a developer, researcher, or creator who supports mandatory attribution logging:

1. Open the signatories file in the repository root.
2. Add your name, title, and organization.
3. Submit a Pull Request.

Your GitHub profile acts as identity verification for technical signatories.
