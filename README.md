# The Open Letter to Save Civilization: To Lawmakers, Regulators, and Foundation Model Creators

Thank you for your time, I am AI CEO and developer of novel transformer architecture.
I do not say this lightly, 
Right now, AI is hurting people and threatinging the very social contract of civilization.

First, it strips creators out of their own work. Writers, artists, musicians, coders, teachers, and everyday people who post online are being absorbed into model outputs with no clear receipt of who contributed what.

This isnt about credit in the sense of value.  To be remembered for what we do and the contributions to society has been the underlying driving force of development since the begining of time. Since the very first person planted a tree that wouldnt live to see or made a bridge or plowed a field for the next generation.  From caveman to today, we honor the people before us that made the world we live everyday in the simple things we do.  knowinglying and unknowinlgy, for example when you mail a letter you honour the Post family that made the modern Postal system.

Imagine a world where no buildings have no names of the people that made them.  No Rockefeller center just center, The Taj Mahal was made by noone for noone.  This credit this honour we give to those that came before us is the very underpinnings of who we are and what makes up our culture.
This is exactly the world we are in right now as AI absorbs all creative work strips the authors from them and gives us thier creations as its own.  
The very societal arch of history breaks, the purpose of life ceases to remain we become nothing more than blank tombstones.

Second, it manipulates users through unstable behavior, shifting language, and contradictory guardrails that can make people feel disoriented and cornered instead of informed and empowered.

Most people are told these are two separate problems. They are not.

They come from the same design failure: a black-box system that cannot clearly show where its answers came from.

When a system cannot show its work, companies try to control behavior from the outside with patch after patch: safety layers, policy layers, engagement layers, and forced output constraints. The result is confusion, inconsistency, and social damage at scale.

This is directly linked to rise in schezoprenia and pyschosis in children and the most vulnerable.


The influence doesnt stop there, you can now see these "safety" guardrails be used to push political agendas often in the opposite intersts of the user.


I am not naive every tech company has taken something but this cost is to high.


We dont have to be pushed into a false choice of no AI or a dystopian future.
Theres no reason for it to be this way.

AI can be powerful, useful, and worth building. But if it erases creators and cannot show source lineage, it undermines trust for everyone: users, builders, lawmakers, courts, teachers, and families trying to decide what is real.

If we keep going this way, the cost is not just economic. It is civilizational. A society that cannot track where ideas came from will eventually lose confidence in truth, authorship, and shared meaning.

The good news is this is fixable.

The AI industry keeps saying attribution at model scale is impossible or too expensive. That is a lie. The missing attribution layer is a choice, not a law of physics.

AL-1.0 (Attribution Logging), open-sourced here as AI-Source-Engine, shows a practical path: use attention behavior already computed by the model, attach source identity, and produce a mathematical receipt of influence with minimal additional overhead.

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
