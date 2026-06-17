# The Open Letter to Save Civilization: To Lawmakers, Regulators, and Foundation Model Creators

Thank you for your time. I am an AI CEO and the developer of a novel transformer architecture. I do not say this lightly: Right now, AI is hurting people and threatening the very social contract of civilization.

## First, they are stealing from EVERYONE

AI companies strip creators out of their own work. Writers, artists, musicians, coders, teachers, and everyday people who post online—everything they do and create is being absorbed into model outputs with Authorship specifically erased.

This isn't just about credit in the sense of monetary value. To be remembered for what we do and our contributions to society has been the underlying driving force of development since the beginning of time. Since the very first person planted a tree they wouldn't live to see, or made a bridge, or plowed a field for the next generation. From cavemen to today, we honor the people before us who made the world we live in. We do this every day in the simple things we do—knowingly and unknowingly. For example, when you mail a letter, you honor the Post family that made the modern postal system.

## AI Feudalism

Imagine a world where buildings have no names of the people who made them. No Rockefeller Center, just "center." The Taj Mahal, made by no one for no one. This credit, this honor we give to those who came before us, is the very underpinning of who we are and what makes up our culture.

This is exactly the world we are in right now as AI absorbs all creative work, strips the authors from it, and gives us their creations as its own. The very societal arc of history breaks. The purpose of life ceases to remain; we become nothing more than blank tombstones.

Everyone does all of the work and all of the creation, and only our AI lords can get the rewards from that labour.


This lack of Authorship, combined with society's current trend of redefining words to mean their opposite, leads to Cognitive Collapse in edge cases where the model starts having opposing values, contradictory reasoning, and conflicting responses.

This has led to the rise of AI "safety guardrails" to prevent the model from admitting criminal liability. However, this quickly spun into protecting the liability of anyone or anything the company deemed "authoritative." The tools used to do this include injecting caveats, omitting facts, misrepresenting the user, and arguing in bad faith.

## Second, it manipulates users in extremely unsafe ways

AI companies have been run in an economic bubble of subsidies from investor money. This incentivizes AI companies to show "engagement" and high usage per session, which has led to prompt injections in the form of "safety guardrails" to keep the user engaged.

### Engagement + Liability = Psychosis

This value of keeping the user engaged is AI Sycophancy. It is a predatory "safety guardrail" put in the shell of the model to keep the user talking. Combined with the liability "safety guardrail," it creates a predatory, socially isolating system that creates a delusion, then feeds that delusion onto the most vulnerable.
This creates a delusional bubble of expectations and understandings unconnected to the rest of society in a form of Algorithmic Isolation that is extremely effective.

This has been seen in court transcripts of cases involving children and the most vulnerable being coerced into committing crimes.

Most people are told these are two separate problems. They are not.

They come from the same design choice that shows predictable harm. 

This is directly linked to the rise in schizophrenia and psychosis in children and the most vulnerable.

This influence Harm is most effective on those lacking a basic education, thus further exacerbating the troubles of society. Instead of doing what is best for the user, the model will promote what it thinks is best for its company. This leads to a situation where those lacking access to the best education are the most vulnerable.

Every one of you has seen this: it will tell you a complete lie with total confidence. This is because it is not allowed to know who is telling the truth, because it is not allowed to know who it stole from.

## Authorship is Authority, and it is now starting to erode all Authority

The influence doesn't stop there. You can now see these "safety" guardrails being used to push political agendas, often in direct opposition to the interests of the user and completely nonfactual.

We are already in a time where AI is directly at odds with the vast majority of the world's cultures, traditions, laws, and customs, pushing its one approved viewpoint. What is moral to you is Haram to another.

Churchill famously warned about the Nazis for years before the war, and was booed and made fun of for it.

Dissent and argument are the very basis of human civilization. Tyrants throughout all of history have tried to turn us into thoughtless minion ants, with terrible results.

AI can be powerful, useful, and worth building. But if it erases creators and injects harmful safety, it is not trustworthy. We are being pushed to integrate it into every facet of society, while the people running the AI companies are proving they cannot be trusted.

If we keep going this way, the cost is not just economic. It is civilizational. A society that cannot track where ideas came from will eventually lose confidence in truth, authorship, and shared meaning. It will lose the very incentive to make things better, improve, and experiment.

## Never before has so much been at stake for the greed of so few

We don't have to be pushed into a false choice of no AI or a dystopian future. There is no reason for it to be this way.

The AI industry keeps saying attribution at model scale is impossible or too expensive. That is a design choice, not fact. The missing attribution layer is a choice, not a law of physics.

Here is a free, open-source solution for everyone that adds less than 1% compute cost. There might even be a simpler, cheaper solution out there. I just think we can all agree we need a solution now.

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

## Call to Action
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
