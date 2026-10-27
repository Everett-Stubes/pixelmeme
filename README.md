# pixelmeme

A 24/7 pixel-art sitcom broadcast, generated live from whatever room it's
sitting in — for a Raspberry Pi 3 mounted vertically. It also runs on
Windows, for testing.

It's an homage to [*Nothing, Forever*](https://en.wikipedia.org/wiki/Nothing,_Forever),
the AI-generated "Seinfeld" livestream that ran on Twitch — same shape (a
fixed sitcom cast, a laugh track, camera cuts, procedurally generated
dialogue, 24/7, no human in the loop), using Jerry, George, Elaine and Kramer
by name, with entirely original jokes rather than real scripts. That one ran
on GPT-3 over the network; this one runs a 1B model locally, which is a
different problem and most of what the writing code is about.

Nothing feeds it and nothing comes out of it but pixels. No microphone, no
speaker, no network: the dialogue is written **on the device** by Llama 3.2 1B
over a lexicon and a conversational-move machine, and it runs forever on a Pi
with nothing plugged into it but a screen. The model is a file on the SD card;
if it is not there the lexicon writes the show on its own.

Personal build for one device in one room — not published, streamed, or
distributed, and not claiming any affiliation with the show.

**ELAINE** says **A GUY FROM MY BUILDING RETURNED THE BIG SALAD.**
**JERRY** says **SINCE WHEN.** It goes on like that and it does not repeat
itself. Every so often — a scheduled segment, not a reaction to anything —
the broadcast cuts away to a looping shot of a man free soloing a mountain,
for a minute, then cuts back.

It opens on the climbing scene for about 25 seconds
(`cutaway.opening_seconds`), then cuts to the apartment.

Built to be left on: no screensaver, no blanking, no idle timeout, and a
crash in one frame does not take the broadcast down.

---

## How it works

The sitcom is **actual 3D** — `render/r3d.py` is a small software rasteriser:
a perspective camera, flat-shaded convex polygons, and the painter's
algorithm. No z-buffer, no textures, no shader beyond one lambert term
against a single directional light. That is not a limitation worked around,
it is the renderer a 1996 console had, and it is why the output looks like
this.

**Nothing is a cube.** Besides boxes there are frusta (a torso narrowing to
a waist, a jaw under a skull), tapered n-sided tubes (limbs, table legs, a
mic stand), wedges (a shoe toe) and **surfaces of revolution** built from a
profile — a skull dome, a lampshade, a stool, a mass of hair. A revolve can
also be an open band, which is what a horseshoe of hair round a bald head
is, and what the sides of a curly mop are.

**And they turn.** Everything an actor emits goes in under a yaw about their
own vertical axis, so they stand at an angle to the lens and face whoever is
talking. Four people square to the camera is a police lineup, and a head
that only ever shows the same facet cannot read as a solid however many
facets it has.

Solids are wound outward and their back faces are **culled** before anything
is filled — about half of every closed shape. Getting that backwards does
not look like a hole: culling then keeps the far side of a dome and drops
the near side, so a skull drew straight through the hair covering it and
everybody went quietly bald. `tools/check_depth.py` checks the winding of
every primitive, permanently.

Three lights, not one: a key from the front-left and well off vertical, a
weak fill from behind and opposite, and a **bounce off the floor**. With a
single key every plane turned away from it landed on flat ambient, so a back
wall and the side wall meeting it came out the same value and the corner of
the room disappeared — and every ceiling was a band of near-black across the
top of the frame.

**Aerial perspective.** Each set declares a fog colour and range, and faces
are mixed toward it with distance. It is one numpy lerp, and without it the
far wall is lit exactly as brightly as the chair in front of it, so the room
reads as a painted backdrop however much furniture is standing in it.

Roughly 1230 polygons a frame emitted, ~650 drawn, 6.5 ms on a desktop,
against 1114 / 560 / 5.1 ms before this pass: a quarter more time for a
third more geometry, three lights instead of one, aerial perspective, and
rooms that are rooms.

It renders into a **108 × 192 portrait surface** which is then blown up to
the panel with nearest-neighbour scaling and mapped onto a fixed palette.
The climbing cutaway is still 2D sprite work — a different programme, a
different look, deliberately.

```
llama 3.2 1B ─┐
              ├─► director ─► stage ─► 108×192
lexicon ──────┘    (beats)   (set +     │
 + move machine               cast)     ▼
                             palette + nearest-neighbour
                                                  │
                                                  ▼
                                             1080×1920
```

### The cast

Four fixed leads, not a randomised pool — a recognisable recurring cast is
the point, the same way the real show never recast its leads. Built for one
device in one room, not published or distributed.

180 polygons each in a wide, 334 in a close-up. **Detail follows the lens**:
the stage knows how far away the camera is and asks for one of three levels,
because sclera and nostrils on a head nine pixels tall are polygons spent on
something nobody can resolve, and they are the polygons the close-up needs.
Only the nearest actor ever gets the top level — in a close the whole cast
can fall inside two metres, and four full-detail bodies made those frames
half as expensive again as every other frame, which shows up as a stutter
rather than as a lower frame rate.

**The head** is a revolved cranium over a revolved jaw. A head built as one
prism has a flat top and a hard rim all the way round, and that rim is the
first thing that says "cylinder with the corners knocked off" — which
matters, because it is what the viewer is looking at.

**The face is built, not drawn on.** A brow ridge, a bridge, a nose tip and
a pair of lips stand proud of the skull and catch the key light on one side
while shading the other. Painted flat on the front facet — a dark bar for a
mouth and two dark squares for eyes — they read as a decal on a tube at any
distance. Close up there are also eye sockets, sclera, pupils, lids,
nostrils and ears.

What stops four people of similar colouring being the same person:

| | |
|---|---|
| **JERRY** | Tallish and lean. Dark hair in a cap with a quiff at the front, and sideburns. Blue shirt with a button placket, white sneakers. |
| **GEORGE** | Short and **round**, not wide — a stocky build given as shoulder width made his torso as broad as it was tall, which is a wardrobe. **Bald**, with a horseshoe of hair from temple to temple round the back, and **round wire glasses**: as one black bar across the face he was wearing a censor strip. |
| **ELAINE** | Big dark **curly hair**, half a head proud of the crown and wider than her face at the jaw. Built as one closed mass it swallowed her face, so the part that falls past the jaw is an open band. A cream blouse under a plum vest, because in one dark top she read as a silhouette with a haircut. |
| **KRAMER** | Very tall and lanky, hair swept **up and back** off the forehead — stacked straight up it was a wizard hat. The loud striped shirt is the whole point of him, so nothing goes over it, and the stripes are stacked frusta: bands pasted on the front of a plain shirt stood out past the body at the waist and poked through both arms. |

Also: a sloped yoke rather than a flat-topped torso with a bar laid across
it; a deltoid bridging torso to arm; hands that continue down the forearm
axis so they cannot detach from the wrist; a stride along the facing axis
with the trailing foot leaving the ground.

The pose is a handful of joint positions from a phase clock and the geometry
is rebuilt every frame; there is no rig and no interpolation.

| Name | Does |
|---|---|
| **JERRY** | Cold-open monologues at the mic, "what's the deal with X" |
| **GEORGE** | Spirals to himself about something nobody else noticed |
| **ELAINE** | Shuts down whatever Jerry or George just said |
| **KRAMER** | Bursts through the door with a scheme |

### The sets

Four, all built as geometry at runtime — nothing is a loaded asset. One world
unit is roughly a metre; the floor is y=0 and the camera sits out past the
missing fourth wall.

- **APARTMENT** — home base. Couch under the window, kitchen counter running
  away down the left wall, fridge, front door back-right, bike on the wall,
  floorboards, cornice, baseboard and a shade hanging from the ceiling.
- **STANDUP** — stool, mic, a pool of light, a stage with a lip, a lighting
  bar overhead, and four heads in silhouette between the lens and the stage.
  Cold opens.
- **DINER** — chequerboard floor, counter and stools, three booths down the
  left wall at three depths, street window, pendants at three depths.
- **HALLWAY** — the building corridor, doors receding down both walls, a
  runner, strip lights, a radiator.

**Each of them has two dressings**, chosen off the seed the stage draws
fresh on every move, so coming back somewhere is coming back at a different
hour rather than cutting to the identical frame it left. The apartment is
daylight or lamplight — cooler and brighter, or warm and a stop and a half
down with the shade lit and the airshaft black with other people's windows
burning in it. The diner is the same two. The club is the brick room or the
one with the red curtain behind the mic, which are both real comedy clubs
and look nothing like each other. The corridor has every strip light on, or
half of them dead.

The show also *moves*. It used to change location only by malfunctioning —
a glitch card and a hard cut — and the stand-up was one-way, because
nothing ever said to come back from it. The brick wall ended up being 45%
of the running time and the apartment 17%, which is the wrong way round for
this show and is most of why a stretch of it looked like one background.
Now the stand-up is a cut-away that returns to wherever everyone was, and
there is an ordinary quiet move alongside the glitch. Measured over three
and a half hours: the three main sets take 28–35% each, the club 4%, a
location lasts about fifty seconds, and **every five-minute stretch
contains at least two of them** (3.2 on average).

The rooms are narrow on purpose, and narrower than they used to be. The frame
is portrait, so the **horizontal** field of view is about 33°: at the distance
a person has to stand to fit head-to-knee in shot, the lens sees about three
metres of width. Walls five metres apart are not spacious, they are
*invisible* — they never enter the frustum at all, and the room collapses
into one flat slab of back wall with the furniture pinned to it. Bringing
them in to about two metres is what puts a corner in shot.

Depth, in order of how much it does:

1. **Things that run away from the lens.** Floorboard seams, a cornice, a
   baseboard, a counter, a row of booths, a line of pendant lamps. Parallel
   lines converging is the whole of perspective; a room with nothing running
   in z is a painting no matter how much is standing in it. The diner floor
   used to be tiled in squares, which produces no edge that converges at all.
2. **Aerial perspective**, as above.
3. **Staging in z**, so the perspective divide gives things visibly
   different scales.

Floors, ceilings and side walls are cut into bands across depth. A face that
crosses the near plane is dropped whole rather than clipped, so one big quad
running from the back of the room to past the lens is simply not drawn — that
was a band of raw void across the bottom of every wide and down the edge of
the hallway, up to a third of the frame. `tools/check_depth.py` renders all four
sets from all nine framings and fails if any of them shows void.

**The camera cuts** rather than pans: a wide that establishes the room, a
medium on whoever is talking, and about one speaker cut in three is a
**close**. A medium puts a head sixteen pixels tall, which is a silhouette;
the face only gets to act in the close.

### The director

Keeps a queue of *beats* — one line, one caption, or a pause — and tops it up
forever. Beats come in **bits**: a petty two-line dispute, an argument that
escalates into static, Jerry's cold open, Kramer's scheme, George's spiral, a
ritual where everyone repeats a word, a broadcast malfunction that cuts the
set. **Most beats have nobody saying anything at all** — someone drifts,
fidgets, or just stands there while nothing renders in the subtitle bar. A
room that narrates itself constantly is the thing this is built to not be.

**The subject is drawn before the conversation is written.** A scene gets a
thing, a culprit, a place and a grievance, and everything in it then refers
to those, which is why six consecutive lines read as a conversation rather
than as six captions that happen to be adjacent.

**The audience is always there.** Every exchange has a button and the button
gets a laugh, plus a scattering of laughs before it — about seven a minute.
It is a caption and only a caption: nothing in this project makes or receives
a sound. Laughs ride over the picture rather than taking a beat of their own,
so one can still be going when the next line starts.

**One running joke per stream.** Early on, the director generates a
catchphrase, realises it once, and freezes the result — it is never
rewritten. That exact sentence then resurfaces every few minutes for the rest
of the broadcast, the "not that there's anything wrong with that" mechanism.

Measured over a simulated hour: about 500 lines, no sentence repeated more
than a handful of times, roughly a quarter of runtime with no dialogue on
screen at all, all four sets visited, and the catchphrase landing two or
three times.

### The writer: a 1B model on the device

Everything above is the **floor**, and it is still here. In front of it is
**Llama 3.2 1B Instruct**, a GGUF quantisation read through llama-cpp-python,
asked for one short exchange at a time on a worker thread. A 1B model fits a
Pi 4 next to a framebuffer and writes three lines in the time the queue takes
to drain, which is the whole reason it is 1B and not 3B. On a **1GB Pi 3**,
where it does not fit, the same code runs Qwen2.5-0.5B instead — see
[Running the writer on a Raspberry Pi 3](#running-the-writer-on-a-raspberry-pi-3).

A model this small is wrong about everything unless you take the decisions
away from it. Three things do that:

**The grammar.** llama.cpp constrains sampling to a GBNF grammar, so
`NAME: LINE`, two to four lines, from this cast, **two to ten words**, ending
on a full stop, question mark or exclamation, in the alphabet the pixel font
can draw, is not a request in a prompt — it is the only thing the sampler is
able to emit. That deletes, before generation rather than after, every
failure that actually costs something: "Sure! Here's an exchange:", markdown
headings, stage directions, a paragraph of narration, a fifth character, and
the curly quotes and em-dashes that this hand-drawn 3×5 font renders as
**solid boxes**.

"Short. Under twelve words." was in the prompt first and was simply not
obeyed. Once it moved into the grammar it could not be ignored — and the
difference was not subtle: unbounded, a turn following a forced one ran to
the token limit as *"MY BOSS HAD TO GO BY THE CLOCK. KRAMER. ELAINE. JERRY.
ELAINE. KRAMER."*

**A grammar that fails to parse does not raise.** `LlamaGrammar.from_string`
writes `error parsing grammar` to the *C library's* stderr — not Python's —
and hands back an object that constrains nothing at all. One escaped hyphen
(`\-`, which llama.cpp's parser has no escape for) therefore turned every
guarantee above off silently, and the output still looked fine, because the
few-shot examples hold the format on their own most of the time. It cost two
rounds of measuring the wrong thing. The enforcement is now verified
functionally in `tools/check_llm.py` rather than by "did it throw".

**The examples.** At 1B, instructions are much weaker than demonstrations.
The register comes from few-shot pairs — the **same** pairs every call, so
the prompt prefix never changes and llama.cpp can reuse it (see the Pi
section; this is the single biggest speed knob there). They used to rotate,
to stop the model reciting one back; at 0.75 it does not, and the filter
drops copies anyway.

Temperature is **0.75**, measured rather than chosen: at 0.9 this model wrote
*"THEY ARE DOUBTFIRED IN THEIR OWN SERVICES"*, at 1.05 it rambled, and 0.75
stayed on the subject while still producing twenty unique lines out of
twenty — so the obvious worry, that a low temperature would make it recite
the examples, turned out not to happen at all.

**The filter.** Output is still untrusted input. It is re-cased, stripped to
the drawable alphabet, cut to the subtitle width on a sentence boundary where
there is one, checked against the last lines actually spoken, and thrown away
whole unless two usable turns survive. A speaker outside the cast is dropped
rather than mapped, because the subtitle bar colours the name from the actor.

Each line is bounded at **twelve words** by the grammar, not by the prompt.
Ten was too tight, and the symptom was strange: a model that had planned a
longer sentence met the cap by *welding the last two words into one* —
"MORE THANYOURSELF", "A BAGEL OF MYOWN", "DROPPING BY FOR THEACK" — and
every one of those lines was exactly ten words long. A non-word on screen
reads as a rendering fault rather than as a joke. At twelve the budget is
about seventy characters, which is where the filter trims anyway, so the
guillotine belongs to the filter: it cuts at a word boundary.

Lines are cut rather than rejected, because a 1B model rambles — and the cut
drops a trailing word that cannot end a sentence, so an over-long line comes
out as *"I HAVE SAID SO REPEATEDLY."* rather than *"...SAID SO REPEATEDLY
FOR."*, which reads as a transmission fault in a show that already has one.

The model gets the scene's facts **and the last two turns actually spoken**.
Without that it opens cold on the same subject every time, which is the exact
failure the lexicon generator was rewritten to stop doing — a language model
has no `Exchange` of its own to consult.

Nothing waits for it. Generation is slower than the queue drains on a small
board, and that is not a stall: it is a stretch of lexicon dialogue. The
floor is not a fallback for a broken model, it is what covers every line the
model has not finished writing yet.

**How much of the show it actually writes.** On a desktop one exchange is
**1.5–5 seconds** for three to six lines, several times faster than the
broadcast consumes them. Measured over 600 seconds of real-time broadcast:
**108 of 109 spoken lines — 99.1%**, 82 generations, 2 rejected, nothing
starved, 19.6 fps held and no frame errors, with the director still drawing
subjects and moving the show between rooms on its own. The one line it did
not write was the catchphrase, which is the one it is not supposed to.

Getting there went wrong twice first, in opposite directions.

**The first attempt starved the lexicon**, and it worked: 100% of the lines,
and a frozen show. `_next_bit` was called *zero times* in 100 seconds — one
subject, one room, for the whole stretch — because everything that makes
this a programme rather than a chat window lives in that function: drawing a
new subject, counting toward a set change, the silent beats, the stand-up,
the glitch. A model fast enough to keep the queue full meant none of it ever
ran. Hence `supply()`: written exchanges are a *supply the director draws
on*, never beats spliced into the queue ahead of it. The director stays in
charge and the model writes the lines inside a bit.

**The second attempt tried to buy the share with frame rate**, and that was
measuring the wrong thing entirely. Attributing every spoken line to what
wrote it says why:

```
written exchanges: 18 served, 0 starved (model had nothing ready)
payloads: 82 supplied, 18 used, 60 dropped unused
```

Nothing was ever starved, and **nearly three quarters of what the model
wrote was thrown away** — the broadcast has nowhere near enough airtime for it. The
share was never supply-limited, so no amount of frame rate could move it. It
was *structural*: the last lines the lexicon wrote were the bits it still
owned outright. Each one had to be handed over on its own terms.

- **Stand-up.** The last holdout, on the reasoning that a four-way exchange
  remapped onto one mouth is not a routine. True of the remapping, false of
  the conclusion: tell the writer it is writing a set, and put **one name in
  the grammar**. There is no second speaker to answer, so a conversation is
  unreachable rather than discouraged.
- **The slip** — one line that does not belong, which nobody acknowledges.
  The director used to insert it *after* the model had finished, which made
  every slip a lexicon line in the middle of a written exchange. The model
  is now offered the same curated pool and writes the turns around its
  choice, so the room ignores it on purpose rather than by accident.
- **The ritual** is the room chanting the subject back at itself. There is
  nothing in it for a model to write — the content *is* the subject, which
  the director already owns — so with a writer attached it is three lexicon
  lines at a time and nothing else. Weight 1; the show does not miss it.
- **The cold open.** `Stage.__init__` advances a beat before `main` has
  finished choosing a writer, so the first exchange was always the
  lexicon's. Now the show stages business and waits up to
  `WARMUP_SECONDS` — a room with people quietly doing nothing in it is a
  perfectly good first ten seconds of a sitcom. Bounded, because the
  alternative is silence for ever if the model never answers — and bounded
  **in bits as well as seconds**, because the clock is the wrong unit on its
  own. The director runs on virtual time: a test or a fast-forward gets
  through thousands of beats in under a real second, so a wall-clock-only
  warmup never expired there and the show staged business for ever. Four
  bits is six to twenty-five seconds of airtime, so on a real broadcast the
  clock still runs out first and the second bound never fires.

What stays lexicon, deliberately: **the catchphrase**. It is realised once
and then repeated verbatim, and a catchphrase the writer rewrites every time
is not a catchphrase.

**Measure it in real time or not at all.** The pacing floor is wall-clock, so
a test that compresses 150 virtual seconds into 38 real ones gives the model
a quarter of the chances it would really get. Measured that way the share
looked like 56% when it was actually 100% — the compressed harness, not the
writer, was what was being measured.

On a Pi 3 the model is the limit rather than the floor, and it will not hold
a share like this. That is what the lexicon is for.

`tools/check_llm.py` holds this up. The filter half runs with nothing
installed — fixed samples of what small models actually get wrong, asserting
what survives — and the model half generates against real scenes and prints
every line, so the register can be read instead of assumed.

### Occasionally, the other thing

About **one exchange in five** is not an argument about a bagel. The same
cast, in the same flat voice, says one thing far too large for the room —
about time, or about being the only person who is really here — and nobody
picks it up. The next line carries on with the small thing.

**It is not prompted. It is forced.** Asking for it did not work: across two
rounds — a paragraph of instruction, then dedicated few-shot examples — the
model marked exchanges existential and then wrote *"MY BOSS SAID THAT WOULD
BE A DISASTER"*, scoring **0 out of 10** both times on whether the exchange
contained anything large at all. A 1B model will not hold two registers at
once, and no quantity of adjectives changes that.

So the large line is drawn from the curated `existential` pool in
`dialogue.py`, realised against the scene that is on now, and the grammar for
a drift exchange is literally `root ::= line big line` where `big` can only
be one of those lines. The model still chooses which one fits, and still
writes the small line before and the deflection after — which is what it is
actually good at. That measures **8 out of 8**, and it has to: the line is
forced, so anything less means the grammar is not being enforced, which is
how the escaped-hyphen bug above was eventually caught.

The deflection is **required**, not optional. The grammar guarantees three
turns, but the filter can still drop the last one, and a large line with
nothing after it is the register without the thing that makes it bearable.
If the deflection does not survive, the whole exchange is dropped and the
lexicon writes that one instead.

Those exchanges get **no laugh track** and are followed by a held silence:
the audience not knowing what to do with it is the joke, and a laugh over it
is the machine explaining itself. That is the same rule the **slip** already
follows, below.

The lexicon generator has its own version of this and always did — an
`existential` register with one deflection after it, held back until the
third turn, because a man staring into the middle distance before anybody has
said what the argument is about is not a tonal shift, it is a non-sequitur.
So the weird register survives on a board too small for the model; what the
model adds is that it is a different line every time rather than one of
twenty.

### The register

Petty, specific, and extremely pleased with itself.

**Nothing is a written line.** `pixelmeme/brain/dialogue.py` holds a lexicon —
words, not sentences — a set of sentence skeletons for each kind of
conversational move, and a state machine that says which moves can follow
which. An exchange is produced by walking that machine: an opener is probed,
a probe is answered, an answer is denied or escalated, and it runs out into a
verdict or a shrug. The same six-turn shape comes out as a different
conversation every time.

What makes the turns connect is that the generator keeps a model of the
**conversation**, not of the last line. An `Exchange` records who is accused,
who is accusing, which question is on the table and who asked it, which
referents have actually been introduced, and what has been established. Every
skeleton declares what it asks for, what it supplies, who is allowed to say
it, what must already be true before it can be said, and which referents it
may introduce. Choosing a line is then a constraint filter — role, then
preconditions, then referents, then answering the open question — and novelty
is the last tiebreak, never a reason to say something impossible.

Measured over 3,000 generated scenes:

| | before | now |
|---|---|---|
| scenes that never say their own subject | 15.7% | **0%** |
| scenes that stop mid-argument instead of ending | 25.2% | **0%** |
| denials with nothing to deny | 4.9% | **0%** |
| the accused becoming the accuser mid-scene | 7.3% | **0%** |
| somebody answering a question put to someone else | 5.2% | **0.4%** |
| questions asked and then dropped | 2.8% | **0%** |
| a person or place carried over from the scene before | 0% | **46%** |

The last row is most of why it feels like one evening rather than six hundred
unrelated clips an hour: a new scene inherits the culprit or the place from
the one before it, and rotates the rest.

Two smaller things do a lot of work. The subject is named once and is "it"
afterwards — repeating the full noun phrase every turn is the loudest tell
that a machine wrote something. And a reply can quote the last few words
back, knowing whose they were: *"I may have mentioned it. In passing." /
"Nothing you do is in passing."* The quote has to be a *fragment* of the
line, not the whole of it; unchecked, "I would remember it." came back as
"I would remember it. Since when is that a thing", which is not somebody
picking up a phrase, it is the machine repeating itself.

**Four rules decide whether a line is allowed to follow the one before it.**

- **Nothing contradicts itself.** Whoever admitted it cannot then deny it.
  `admitted` was recorded and never read, so the person who had just said
  "it's not like I planned it" would follow it with "you're thinking of
  somebody else" — the one thing in a conversation a listener will not
  forgive.
- **Nobody follows their own question.** Barring only a *third* turn
  running let somebody ask "what about it." and then answer it themselves
  on the very next line. (Two turns running is fine and often right: an
  accuser pressing and then delivering the verdict is one person's beat.)
- **Nothing new arrives at the end.** A person or a place can be put on the
  table while the scene is being set up, not as the closing line. Hearing
  about somebody's brother-in-law for the first time in the last breath of
  a scene is a new character walking in during the credits.
- **Come back to it.** A line that names the subject, answers the question
  on the table, quotes the line before it, asks one of its own, or presses
  an accusation is joined to the conversation. One that does none of those
  is a new sentence about a new thing, and the next line has to come back.
  A third voice cutting in is held to this always — it is the one place a
  non-sequitur is least forgivable.

Measured over 4,000 scenes, with the rules switched off at runtime so both
columns are the same measurement on the same scenes:

| | rules off | rules on |
|---|---|---|
| adjacent pairs that follow | 85.8% | **91.2%** |
| somebody denying what they admitted | 0.8%¹ | **0%** |
| a person or place named first in the last two turns | 17.6% | **0%** |
| three turns running touching nothing on the table | 1.3% | **0.1%** |
| direct questions answered by the very next line | — | **81%** |

¹ measured before the rule went in; it is enforced in two places and only
one of them can be switched off from outside.

The existential register also has to wait now: three turns in is a man
staring into the middle distance before anybody has finished saying what
the argument is about.

**Occasionally somebody says something true.** About one scene in twenty-five
lands on a line like *"we're all just people who haven't been found out yet"*
or *"somebody will say my name for the last time and not know it."* It gets
exactly one deflection — *"anyway."*, *"do you want a glass of water."* — and
then the scene goes back to being about a parking spot, which is the only way
a line like that is bearable in a sitcom.

Roughly one exchange in five has a **slip** dropped into the middle of it —
a line that does not belong, which nobody acknowledges, and the conversation
carries on. Unremarked is what makes it land; a slip with a reaction is just
a punchline. Slips never get a laugh. The silence after one is the joke.

### It works offline

There is nothing to be online for. The writer is a **1B model on the device**
and the floor under it is a grammar, so after the one-time model download
nothing talks to the network: the sets, the cast and the climb are all built
out of numbers at runtime, and the dialogue is written either by a GGUF file
on the SD card or by `pixelmeme/brain/dialogue.py`.

If you'd rather have Claude write it over the network, see
[Cloud script](#cloud-script-optional). Off by default.

### It degrades instead of failing

| If this is missing | What happens |
|---|---|
| The baked idle clip | A pixel-art climb is generated procedurally |
| The GGUF model, or llama-cpp-python | The lexicon generator, silently |
| Enough RAM to hold the model | The next model in the list is tried; if none fit, the load is refused rather than attempted |
| A llama-cpp-python wheel that runs | `tools/probe_llm.py` catches it at install time, not at runtime |
| The network / API key | Nothing; only the optional cloud script wants it |

---

## Running the writer on a Raspberry Pi 3

The 1GB Pi 3 is the board this project targets, and it is the one a language
model least wants to run on. It does work, with a different model and one
important change to the prompt.

**Memory.** Measured on a desktop, the whole process is **52MB** without a
model and **958MB** with Llama 3.2 1B Q4 loaded and generating — the 808MB
file plus about 150MB of context and overhead. A Pi 3 has 1024MB before the
GPU split and the OS, so the 1B is not close. The model that fits is
**Qwen2.5-0.5B-Instruct Q4_K_M, 398MB**, leaving the board about 150MB spare
on a console boot.

```bash
python tools/get_model.py --auto      # works it out from free memory
python tools/get_model.py pi          # or name it
```

`install.sh` runs `--auto`, so a fresh Pi install gets the right one without
being told which board it is on.

**A 0.5B at Q4 beats a 1B at Q2.** Llama 3.2 1B does come in a 581MB Q2_K
that would nearly fit, but Q2 on a model this small degrades it badly and
581MB leaves almost nothing spare. The 0.5B is the better trade, and it is
honestly weaker than the 1B either way — expect more malformed words
("BAGLE", "SAYSING"), more repetition, and the odd line that is simply wrong
("THERE ARE NOTHINGS FOR ME TO DO."). The lexicon generator is underneath it
and will write most of the show on this board.

**The prompt matters more than the model here.** The prompt is ~530 tokens of
few-shot examples, and llama.cpp can reuse a prefix it has already evaluated
— but only if the prefix does not change. The examples used to be rotated per
call, which defeated that entirely. Measured on the 1B:

| | call 1 | call 2 | call 3 |
|---|---|---|---|
| examples rotated | 4.5s | 4.3s | 4.9s |
| **fixed (`llm.stable_prompt`)** | 6.4s | **3.2s** | **2.7s** |

On a desktop that is two seconds. On a 1.2GHz A53, where evaluating the
prompt is most of the work, it is the difference between a model that
contributes and one that never finishes a line. Rotation existed to stop the
model reciting an example back; at temperature 0.75 it does not, and the
filter drops copies anyway.

**Also worth doing on the Pi:** use the 64-bit Raspberry Pi OS, boot to a
console rather than a desktop (the desktop session's ~250MB is the difference
between fitting and not), and leave `threads: 0` so it takes three of the
four cores and leaves one for the renderer.

**Measure it on the Pi itself.** Nothing here can tell you how fast a Pi 3
actually generates — it is a different CPU by a factor of ten. This prints
the real numbers on whatever board it is run on:

```bash
python tools/check_llm.py --n 6
```

If the mean latency is far above the twenty seconds or so the beat queue
takes to drain, that is not a fault: the lexicon covers the gap and the model
contributes what it can. `--no-llm` turns it off entirely.

---

## The climbing cutaway

After about **five scenes** (a scene being one continuous stretch on one
set), the broadcast cuts to a black card — `MEANWHILE...` — shows a man free
soloing a mountain, then cuts back with `AND NOW, BACK TO OUR REGULARLY
SCHEDULED NOTHING`. It is a **scheduled segment**, not a reaction to a quiet
anything. It runs on a scene count and a clock, nothing else.

Five scenes runs about **four minutes**, with a floor and ceiling in seconds
(`cutaway.min_live_seconds` / `max_live_seconds`) so a run of very short
scenes cannot cut away constantly and a very long one still breaks. The floor
has to stay *below* what `every_scenes` takes to play, or the floor is the
only thing that ever decides and the cut lands in the middle of a scene
instead of at the end of one.

**The break has to be shorter than the programme.** It used to be five
minutes against a two-and-a-half-minute sitcom floor — 62% of the running
time was the climbing wall — and because the opening climb is only
twenty-five seconds, the first real break was *twelve times longer* than the
one the viewer had already been shown. That does not read as a station break,
it reads as the show never coming back; one was reported as exactly that. At
a one-minute climb it is about **four fifths sitcom**, and the break is
shorter than the stretch of show on either side of it.

`tools/check_schedule.py` drives the real scheduler over three virtual hours
and asserts the shape rather than the numbers: both segments play, the sitcom
comes back dozens of times, no stretch outruns its guard rails, the opener is
the short one, every real break is the same length, **the break is shorter
than the programme**, and the sitcom holds at least 70% of the running time.
It currently measures 81%, with sitcom stretches of 200–300 s against a
60-second break.

The schedule is also the first thing each frame, before anything that can
throw. The main loop swallows a frame that raises, so that the installation
cannot be taken down by one bad beat — which means everything after the throw
is skipped, and with the clocks further down a permanently broken frame would
pin the broadcast inside whichever segment it was in. A stuck sitcom is a
black screen; a stuck cutaway is a climb that never ends.

Drawn in a 3/4 angle (the near-side limbs draw last and full thickness, the
far side first and thinner, which is what fakes the turn at this resolution),
red shirt, black hair, chalk bag at the hip. The likeness is three things: a
jaw as wide as the skull above it with a chin hung off the front, ears that
stand off the sides, and a swimmer's V of shoulders over a narrow waist.
Nothing is overlaid on the shot — it explains itself.

**He does more than one move.** A sequence strings together a reach, a high
step, a press out onto that foot, a one-armed shake-out, a match and a
settle, each with its own duration and its own sense of what follows it — a
step wants a press on it, a press leaves him wanting a hand.

**The wall is scrolled by exactly the height he just climbed**, and by
nothing else. Each move declares a height in body units rather than a
speed, the height is paid out on the same eased curve the limbs move on,
and the caller scrolls the rock by the change in one running total — so
there is only one number and the two cannot drift. Given as a speed per
move, the rock slid past at a constant rate through a press while he
accelerated and decelerated under it, and kept sliding during a shake-out
when he was not moving at all; he looked like he was swimming up a texture.
A press is now a rush of rock and a shake-out is a dead stop, which is
about a fifth of the time. He also no longer rises up the screen during a
press and snap back at the end of it: the camera follows him, so gaining
height moves the *wall*. The press reads from the legs straightening and
the holds coming down past his hands, which is what actually happens.

**Nothing in the scene is dithered**, because nothing in it is a gradient.
The eight sandstone tones and the climber's own colours are palette entries
to the value, taken from the list rather than picked by eye — picked by eye
all eight missed by two to twelve per channel, which left the quantiser
choosing between the two entries each one fell between, and an ordered
dither laid that out as a grid of dots across whole beds of rock.
`tools/check_flat.py` renders both the sitcom and the climb and fails if
any large region of one colour comes out as more than one.

He is always **ascending**: the rock scrolls down the screen, new stone
entering at the top, which is what reads as the camera climbing rather than
sinking. The pillar is **not a rectangle** — its silhouette is cut by a
profile built from whole-number harmonics over the tile height, so the edges
wander, pinch and bulge, with square-edged flakes bitten out of them, and the
cut still wraps exactly when the tile does.

**It used to look like a wooden post**, and the reason was regularity in one
axis: nine evenly spaced bedding planes down the tile, and a crack generator
that ran *horizontally* with a small vertical wobble. Warm brown, horizontal
grain, wavy dark lines along it — that is timber, whatever the palette is
called. The beds are now three harmonics beaten against each other so no two
are the same thickness, and the strongest feature on the face runs the other
way: **desert varnish**, dark mineral streaks weeping vertically down the
rock, which is what sandstone actually looks like from a distance. Cracks run
up it now rather than across, taper out at both ends instead of being cut,
and there are three rather than ten — at ten, crossing the ledges, the pillar
read as scaffolding, which is a worse failure than the plank it replaced.

Some of the pockets are **chalked**, clustered into a corridor up the middle
rather than scattered: a route is a line, and chalk spread evenly over a face
is just white speckle. It is the one thing on the wall drawn off the
sandstone ramp, because it is magnesium carbonate on rock and not a lighter
rock.

The panorama either side is built **in depth layers, each drawn at the
contrast its distance earns** — a far range nearly the colour of the sky, a
near range carrying the snow, a valley floor split into a lit far half and a
shaded near one, and a village that is four or five marks rather than a
terrace of chalets with windows. The eye reads "far" from loss of contrast
long before it reads it from size, and the old backdrop drew everything at
full contrast, so the climb looked twenty feet off the ground no matter how
long it scrolled. Three **birds track across the valley below him**, which is
the cheapest altitude cue there is: a bird above a figure says nothing, the
same bird seen from above says the ground is a long way down. The pillar was
also narrowed from 78% of the frame to 62%, because the drop is the subject
and there was barely any of it visible.

It is all rendered once and held fixed — nothing at that distance would
visibly move during one pitch anyway — and the whole climb costs **1.1 ms a
frame**, about 2% of the budget at 20 fps, with a 7 ms one-off build.

No footage is bundled. Supply your own and bake it, on a desktop:

```bash
pip install opencv-python
python tools/prepare_video.py your-climbing-clip.mp4 --seconds 45
```

That writes `assets/idle/honnold.pxv.npz`, picked up on the next start. Baking
crops to portrait (not squashed), palette-quantises, and stores one byte per
pixel, so playback is one array lookup per frame instead of an H.264 decode.
45s at 12 fps is about 17 MB in RAM; the tool warns you if you ask for more
than a Pi 3 will enjoy.

Until you do, the procedural fallback above runs — it tiles vertically, so it
never repeats and never runs out.

---

## Install (Raspberry Pi 3)

To try it on a Windows PC first, skip to
[Running it on a Windows PC](#running-it-on-a-windows-pc-to-test).

```bash
git clone <this repo> ~/pixelmeme
cd ~/pixelmeme
./install.sh
./run.sh --windowed --debug
```

`install.sh` installs the SDL2 system libraries, makes a virtualenv and
installs the three Python dependencies. Safe to re-run.

It also installs the writer — llama-cpp-python, which compiles from source
here because there is no prebuilt aarch64 wheel, and then **whichever model
fits this board** via `tools/get_model.py --auto`: the 398MB Qwen2.5-0.5B on
a 1GB Pi 3, the 808MB Llama 3.2 1B on a Pi 4. Neither step failing stops the
install, and if both fail the lexicon generator writes the show as it always
did. See [Running the writer on a Raspberry Pi 3](#running-the-writer-on-a-raspberry-pi-3).

The debug overlay shows state, frame rate, the current set and speaker,
polygon count, time to the next cutaway (or time back from one), and, when
the model is writing, its status and a running `OK / rejected /
existential / standup / slip`
count.

### Making the display portrait

Rotate the panel at the OS level — cheaper than rotating every frame. In
`/boot/firmware/config.txt`:

```
display_rotate=1      # 90° clockwise; 3 for anticlockwise
```

Otherwise set `rotate: 90` in `config.yaml` and the app turns each frame.

### Running it on boot

```bash
sudo cp systemd/pixelmeme.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pixelmeme
journalctl -u pixelmeme -f
```

Edit the unit if you cloned somewhere other than `/home/pi/pixelmeme`. It is
`Restart=always` with no start limit, so it comes back from anything.

### Keeping the screen awake

Three places, because one is never enough:

- `run.sh` runs `xset s off`, `xset -dpms`, `xset s noblank` at startup.
- The app re-runs them every 60s (`display.keepalive_seconds`), in case a
  desktop session turns blanking back on underneath it.
- `SDL_VIDEO_ALLOW_SCREENSAVER=0` and `set_allow_screensaver(False)` stop SDL
  idling the display itself.

On a console-only Pi there is no `xset` and nothing is lost — there is no X
screensaver to disable, and SDL uses KMSDRM automatically when `DISPLAY` is
unset.

---

## Running it on a Windows PC (to test)

```powershell
.\setup.ps1                      # venv + dependencies
.\run.ps1 --windowed --debug
```

You do not have to run `setup.ps1` first — `run.ps1` falls back to whatever
`python` is on your PATH and tells you what is missing rather than throwing a
traceback. If PowerShell blocks unsigned scripts, use the double-clickable
`setup.cmd` and `run.cmd` instead.

- **Screen sleep** uses `SetThreadExecutionState` instead of `xset`, and is
  released on exit — testing this will not silently disable your monitor
  timeout.
- **Fullscreen on a landscape monitor** letterboxes the portrait image.
  `--fit fill` uses the full screen height instead; `--windowed` gives a small
  window.
- **Baking the idle clip** is a desktop job anyway — do it here and copy the
  `.npz` to the Pi.

---

## Configuration

Everything is in `config.yaml`, with documented defaults in
`pixelmeme/config.py`.

| Key | Default | Notes |
|---|---|---|
| `display.base_width/height` | 108 × 192 | Exactly 10× on 1080×1920. `90x160` is chunkier and cheaper |
| `display.fps` | 20 | |
| `display.dither` | 0.55 | Only applies to palettes outside the flat set; both default looks are flat |
| `display.palette` | `nyc` | Tonal ramps for flat shading. Also `sitcom`, `vhs`, `fried`, `gameboy`, `stone` |
| `display.dither_flat` | 0 | Must stay 0. Used by both `nyc` and `stone`: an ordered dither on anything meant to be one solid colour checkerboards it |
| `display.fit` | `auto` | `fill` uses the whole screen height |
| `cutaway.every_scenes` | 5 | Cut to the climb after about this many scenes |
| `cutaway.opening_seconds` | 25 | The short climb shown at startup |
| `cutaway.duration_minutes` | 1 | How long each cutaway lasts. Keep it under the live floor — a break longer than the programme reads as the programme never coming back |
| `cutaway.min/max_live_seconds` | 150 / 1500 | Guard rails on the sitcom stretch. The floor must stay below what `every_scenes` takes, or the cut lands mid-scene |
| `llm.enabled` | `true` | The on-device writer. Harmless with no model file — startup says so once and the lexicon covers |
| `llm.model_path` | a **list**: the 1B then the 0.5B | First entry present and small enough wins, so one config fits every board |
| `llm.ram_headroom_mb` | 220 | Spare memory needed on top of the model file. The gate is the file's size plus this, against memory actually free |
| `llm.stable_prompt` | `true` | Fixed few-shot examples, so llama.cpp reuses the prompt prefix. Worth 2s a call on a desktop, far more on a Pi |
| `llm.temperature` | 0.95 | Under about 0.8 a 1B model recites the few-shot examples back |
| `llm.existential_chance` | 0.18 | How often the exchange is the weird one instead of the small one |
| `llm.standup_chance` | 0.20 | Share of calls spent writing a routine instead of an exchange |
| `llm.slip_chance` | 0.20 | Share of calls that carry a slip the model chose and wrote around |
| `llm.min_interval_seconds` | 15 | Floor between generations, not a target |

Flags beat the file: `--windowed`, `--fps`, `--palette`, `--rotate`,
`--fit`, `--base 72x128`, `--debug`, `--cloud`, `--no-llm`,
`--llm-model PATH`.

`--no-llm` is the one to reach for if a board is struggling: it writes from
the lexicon and loads nothing.

### Keys while it is running

| Key | Does |
|---|---|
| `Esc` / `q` | Quit |
| `f` | Toggle fullscreen |
| `d` | Toggle the debug overlay |
| `Space` | Skip the current beat |
| `i` | Force the climbing cutaway on or off |

---

## Cloud script (optional)

A different thing from the on-device model: that one is a file on the SD card,
this one is an API call. If both are on, this one wins — it is the more
deliberate switch of the two. If you want sharper writing than 1B gives you
and the Pi has a network connection:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
./run.sh --cloud
```

- Requests run on a **background thread**; the local generator keeps the
  stream going while one is in flight, so it never waits on the network.
- Claude is given the cast list (Jerry/George/Elaine/Kramer) and returns an
  exchange with a speaker per line, spliced into the front of the queue.
- `claude-opus-5` at `effort: low`, JSON schema, one call per 20s, 12s timeout.
- Every failure — no key, offline, timeout, rate limit, refusal — falls back
  without a visible seam.

This is the only thing in the project that touches the network at runtime,
and the only thing that sends anything anywhere — the scene's own invented
subject. That is why it is off by default. The on-device model also arrives
over the network, once, as a file.

---

## If the writer does not start

The app says which writer it got, once, at startup:

```
[writer] llama: ready
[writer] llama unavailable: no model at .../Llama-3.2-1B-Instruct-Q4_K_M.gguf
[writer] the lexicon generator
```

- **`no model at ...`** — run `python tools/get_model.py`. It resumes, so a
  dropped transfer is not a restart; the first run of it here stopped at 79MB
  of 770MB with no error at all, just a closed socket.
- **`llama-cpp-python not installed`** — `pip install llama-cpp-python`, or
  on Windows take the prebuilt CPU wheel (see `requirements.txt`).
- **`load failed: OSError`, or a crash inside `llama_backend_init`** — the
  wheel installed but does not run on this machine. It happened here with
  **0.3.36**, which died with an access violation; **0.3.2** worked on the
  same box with the same model. `python tools/probe_llm.py` tells you which
  you have in about a second, and both installers run it and pin back on
  failure. Nothing about this is fatal — the show runs off the lexicon
  either way, which is exactly why it is easy not to notice.
- **`... needs 618MB, 430MB free`** — the model on disk does not fit the
  memory actually free. Get a smaller one (`python tools/get_model.py pi`, or
  `tiny`), or free memory up — booting to a console instead of a desktop is
  worth about 250MB on a Pi 3, which is usually the whole difference.

---

## Performance

The full path (build scene → rasterise → effects → quantise → upscale to
1080×1920) is about 6.5 ms per frame measured on a desktop, of which 4.3 ms
is the 3D. A Pi 3 is roughly 5–8× slower, which leaves headroom at 20 fps. If it
stutters:

1. `display.fps: 15`
2. `display.base_width: 90`, `base_height: 160`
3. `display.dither: 0.0`

Memory is a few tens of megabytes, unless you bake an idle clip, in which
case the clip dominates everything else.

**The writer is the exception to all of the above.** Llama 3.2 1B at Q4 is
about 1.3GB resident and it is the only thing here that competes for CPU with
the render loop. Two things keep it off the frame rate: it generates on a
worker thread (llama.cpp releases the GIL, so this is real parallelism, not
cooperative), and it takes **one thread fewer than there are cores**, leaving
one for the renderer. Four threads of matrix multiply against a 20 fps budget
on four cores is a fight the frame loses, and generation being slower is
invisible while a dropped frame is not.

**The memory check is arithmetic, not a constant.** A model loads only if
the memory actually free covers the model file's own size plus
`llm.ram_headroom_mb` (220MB: context, compute buffers, the renderer, the
OS). A fixed floor was wrong in both directions — it refused a 400MB model on
a 1GB board that holds it comfortably, and it would have waved through a 2GB
model on a 2GB board that cannot. Free memory, not total: a Pi 3 has 1GB on
paper, and what is left after the GPU split, the OS and any desktop session
is what decides whether the thing runs or the board spends a week swapping to
an SD card.

`llm.model_path` is therefore a **list**, tried in order, and the first entry
that is both present and small enough wins. One config for every board:

| free memory | picks |
|---|---|
| Pi 3, console (~760MB) | Qwen2.5-0.5B Q4, 398MB |
| Pi 3, with a desktop (~430MB) | nothing — refused, lexicon writes |
| Pi 4 2GB (~1700MB) | Llama 3.2 1B Q4, 808MB |
| this desktop | Llama 3.2 1B Q4 |

---

## Layout

```
pixelmeme/
  main.py            LIVE/IDLE state, cutaway scheduling, main loop, CLI
  config.py          defaults + config.yaml merge
  brain/
    dialogue.py      the lexicon and the move machine -- the floor
    llama.py         Llama 3.2 1B on the device: grammar, prompt, filter
    director.py      scenes, casting, staging, laughs, the runner joke
    cloud.py         optional Claude scriptwriter, over the network
  render/
    stage.py         the continuous shot: beats -> picture
    r3d.py           the software 3D rasteriser
    actors.py        the fixed cast, as box assemblies
    world.py         the four sets, as geometry, plus the cameras
    display.py       window, portrait scaling, screen-blanking defence
    palette.py       16-colour LUT quantiser + ordered dither
    pixelfont.py     hand-drawn 3x5 bitmap font
    effects.py       glitch, chromatic shift, speckle
    laughtrack.py    the audience, as a caption strip
    idle.py          baked-clip player + procedural climbing cutaway
    sprites.py       the climber, his moves, the rock and the backdrop
tools/
  prepare_video.py   bake a clip into the idle format
  make_palette.py    derive the 3D palette from the materials in use
  get_model.py       download the writer (Llama 3.2 1B, GGUF, ~800MB)
  probe_llm.py       does the installed llama-cpp-python actually run here
  check_depth.py     every set from every framing: no void, no bad winding
  check_flat.py      nothing flat comes out checkered
  check_schedule.py  the broadcast always comes back from the cutaway
  check_llm.py       the model's output filter, and the model if it is here
systemd/             run on boot, restart always
install.sh / run.sh              Raspberry Pi
setup.ps1 / run.ps1 (+ .cmd)     Windows
```

**To change the writing, edit `pixelmeme/brain/dialogue.py`.** Add nouns,
verbs, places or verdict words to the banks at the top and everything
downstream picks them up. Add a sentence skeleton to a move and it enters the
rotation wherever that move can occur. Add a whole move, give it entries in
`FOLLOWS`, and the machine starts reaching it.

Turns are short on purpose. `SLIPS` is the schizo seasoning: occasionally one
line that does not belong gets dropped into an exchange and nobody
acknowledges it. You do not need to understand the rest of the program to
make it funnier.
