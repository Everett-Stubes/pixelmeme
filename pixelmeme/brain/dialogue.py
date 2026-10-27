"""The dialogue generator.

Nothing here is a written line. There is a lexicon -- words, not sentences --
a set of sentence skeletons tagged with what they do to a conversation, and a
machine that walks from one to the next. A scene is produced by walking that
machine against a growing record of what has been said, so the same six-turn
shape comes out as a different conversation every time.

The thing that makes generated turns connect is not the writing. It is that
the generator keeps a model of the **conversation** rather than of the last
line. An `Exchange` records who is accused, who is accusing, which question
is on the table and who asked it, which referents have actually been
introduced, and what has been established. Every skeleton declares:

  asks     what it puts a question to -- a place, a time, a name, a count
  gives    what it genuinely supplies, or "~x" for a dodge that acknowledges
           the question and withholds the answer
  role     "A" only the accuser may say it, "D" only the accused, "-" either
  needs    a precondition: an accusation must exist, somebody must have
           denied, or a specific phrase must already be in the transcript
  intro    which referents this line is allowed to introduce for the first
           time; a line that merely refers to a place is illegal until some
           other line has established one
  sets     what saying it establishes: an accusation, or an admission

Selection is then a constraint filter, not a weighted draw from a bucket:
role, then preconditions, then referents, then answering the open question,
and only then novelty. Novelty is the last tiebreak and is never a reason to
say something impossible.

On top of that: the subject is named once and is "it" afterwards, a reply can
quote the last few words back, and once in a while somebody says something
true and nobody is equipped to pick it up.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import NamedTuple

# -- lexicon -----------------------------------------------------------------
# Words. The generator builds every phrase it needs out of these.

NOUNS = [
    "SOUP", "BAGEL", "JACKET", "PARKING SPOT", "REMOTE", "RECEIPT", "SWEATER",
    "COUPON", "THERMOSTAT", "BUFFET", "RESERVATION", "HAIRCUT", "DELIVERY",
    "BIRTHDAY CARD", "WAITING LIST", "GUEST TOWEL", "RETURN POLICY", "MAIL",
    "SPARE KEY", "CAR ALARM", "WINDOW UNIT", "SHOPPING CART", "SIDE DOOR",
    "PARKING TICKET", "ANSWERING MACHINE", "LEFTOVERS", "GUEST LIST", "REFUND",
    "FIRE ESCAPE", "TIP JAR", "DRY CLEANING", "RENTAL CAR", "BIG SALAD",
    "AIR CONDITIONER", "MOVING TRUCK", "PUFFY SHIRT", "SECOND DATE",
]

ADJECTIVES = [
    "SECOND", "LAST", "FREE", "WRONG", "EXTRA", "WHOLE", "SO-CALLED",
    "BACKUP", "ORIGINAL", "REPLACEMENT", "EMERGENCY", "GOOD", "OTHER",
]

# base, past. Every one takes an object, which is what lets the generator
# build a deed out of a verb and the subject without checking anything.
VERBS = [
    ("TAKE", "TOOK"), ("MOVE", "MOVED"), ("RETURN", "RETURNED"),
    ("EAT", "ATE"), ("HIDE", "HID"), ("COUNT", "COUNTED"),
    ("SIGN FOR", "SIGNED FOR"), ("OPEN", "OPENED"), ("WEAR", "WORE"),
    ("SELL", "SOLD"), ("LOSE", "LOST"), ("BORROW", "BORROWED"),
    ("FINISH", "FINISHED"), ("CANCEL", "CANCELLED"), ("REPORT", "REPORTED"),
    ("TOUCH", "TOUCHED"), ("SMELL", "SMELLED"), ("FOLD", "FOLDED"),
    ("STACK", "STACKED"), ("SPLIT", "SPLIT"), ("PROMISE", "PROMISED"),
]

PLACE_NOUNS = [
    "DINER", "LOBBY", "PARKING GARAGE", "DELI", "LAUNDRY ROOM", "ELEVATOR",
    "BANK", "HALLWAY", "MOVIE THEATER", "WAITING ROOM", "BASEMENT",
    "STAIRWELL", "POST OFFICE", "SUPERMARKET", "BACK OFFICE", "SIDE ENTRANCE",
]

RELATIONS = [
    "DENTIST", "COUSIN", "LANDLORD", "SUPER", "MAILMAN", "ACCOUNTANT",
    "NEIGHBOR", "BOSS", "BROTHER-IN-LAW", "OPTOMETRIST", "NIGHT MANAGER",
    "FORMER ROOMMATE", "DOORMAN", "CONTRACTOR", "TRAVEL AGENT",
]

POSSESSIVES = ["MY", "HER", "HIS", "THE", "THEIR"]

# Verdicts are built, not listed: a modifier and an agent noun. The joke in
# the format is that any two of these make a complete and final judgement of
# a human being.
TRAIT_MODS = [
    "CLOSE", "LOW", "SLOW", "DOUBLE", "CONSTANT", "SILENT", "PROFESSIONAL",
    "PART-TIME", "CHRONIC", "SERIAL", "CASUAL", "COMPETITIVE", "SEASONAL",
]
TRAIT_AGENTS = [
    "TALKER", "DIPPER", "BLINKER", "SAVER", "HOVERER", "RE-GIFTER", "CHEWER",
    "NODDER", "WAVER", "STARER", "EXPLAINER", "SHOUTER", "TOUCHER",
    "LINE CUTTER", "NAPKIN HOARDER",
]


# -- phrase building ---------------------------------------------------------


def noun_phrase(rng: random.Random) -> str:
    """"THE SOUP", "THE SECOND BAGEL", "THAT SO-CALLED REFUND"."""
    noun = rng.choice(NOUNS)
    det = "THE" if rng.random() < 0.8 else "THAT"
    if rng.random() < 0.35:
        adj = rng.choice(ADJECTIVES)
        # Not an adjective the noun already opens with. "SECOND" is in both
        # lists and "SECOND DATE" is a noun, so this produced "THE SECOND
        # SECOND DATE" about once in a thousand subjects -- which would be
        # a shrug, except the catchphrase is realised once and then repeated
        # verbatim for the rest of the broadcast. A one-in-a-thousand
        # stutter that lands there is on screen every few minutes for as
        # long as the installation is up.
        if not noun.startswith(adj + " "):
            return f"{det} {adj} {noun}"
    return f"{det} {noun}"


def person(rng: random.Random) -> str:
    """"MY DENTIST", "A GUY FROM THE DELI", "HER COUSIN'S LANDLORD"."""
    roll = rng.random()
    if roll < 0.2:
        return f"A GUY FROM {place(rng)}"
    if roll < 0.32:
        return f"{rng.choice(POSSESSIVES)} {rng.choice(RELATIONS)}'S " \
               f"{rng.choice(RELATIONS)}"
    return f"{rng.choice(POSSESSIVES)} {rng.choice(RELATIONS)}"


def place(rng: random.Random) -> str:
    return f"THE {rng.choice(PLACE_NOUNS)}"


def trait(rng: random.Random) -> str:
    """"LOW TALKER", "SEASONAL RE-GIFTER". No article: skeletons add one
    where they need it, so "THAT'S {TRAIT} BEHAVIOUR" stays grammatical."""
    return f"{rng.choice(TRAIT_MODS)} {rng.choice(TRAIT_AGENTS)}"


# -- a scene -----------------------------------------------------------------


@dataclass
class Scene:
    """What one exchange is about. Drawn once; every line refers to it."""

    thing: str = "THE SOUP"
    # The bare noun inside `thing`, for the places a sentence needs a noun
    # without its determiner -- "A DIFFERENT SOUP", not "A DIFFERENT THE SOUP".
    noun: str = "SOUP"
    guy: str = "MY DENTIST"
    place: str = "THE DINER"
    verb: tuple[str, str] = ("TAKE", "TOOK")
    trait: str = "CLOSE TALKER"
    number: int = 4
    # Which referents have actually been said out loud. Nothing may refer to
    # a place or a person the conversation has not introduced, and the
    # subject becomes a pronoun only after it has been named once.
    said: set[str] = field(default_factory=set, repr=False)

    @property
    def deed(self) -> str:
        return f"{self.verb[1]} {self.thing}"

    def ref(self, rng: random.Random) -> str:
        """The subject, named the first time and pronominalised after."""
        if "thing" not in self.said:
            self.said.add("thing")
            return self.thing
        if rng.random() < 0.15:        # occasionally back to the full name
            return self.thing
        return "IT" if rng.random() < 0.72 else "THAT"


def bare_noun(phrase: str) -> str:
    """"THAT SECOND BAGEL" -> "BAGEL". Determiners and adjectives are all
    single leading words, so dropping known ones is enough."""
    words = phrase.split()
    while len(words) > 1 and words[0] in ("THE", "THAT", "A", "AN", *ADJECTIVES):
        words = words[1:]
    return " ".join(words)


def new_scene(rng: random.Random, thing: str | None = None,
              previous: "Scene | None" = None) -> Scene:
    """Draw a subject. A scene inherits from the one before it where it can.

    Redrawing every noun every twenty seconds is why the stream used to feel
    like a shuffled deck: six hundred subjects an hour, nothing carried
    between any two of them. Keeping the culprit or the place for a few
    scenes is most of what makes it feel like one evening in one building.
    """
    thing = thing or noun_phrase(rng)
    guy = previous.guy if previous is not None and rng.random() < 0.45 \
        else person(rng)
    where = previous.place if previous is not None and rng.random() < 0.40 \
        else place(rng)
    return Scene(
        thing=thing,
        noun=bare_noun(thing),
        guy=guy,
        place=where,
        verb=rng.choice(VERBS),
        trait=trait(rng),
        number=rng.choice((2, 3, 4, 6, 7, 9, 11, 12, 40)),
    )


# -- the shape of a line -----------------------------------------------------


class Line(NamedTuple):
    text: str
    asks: str = ""       # "where" / "when" / "who" / "why" / "many"
    gives: str = ""      # the same, or "~where" for a dodge
    role: str = "-"      # "A" accuser only, "D" accused only, "-" either
    needs: str = ""      # "accused" / "denied" / "notfirst" / "said:PHRASE"
    intro: str = ""      # referents this line may introduce
    sets: str = ""       # "accuse" / "admit"


def L(text, asks="", gives="", role="-", needs="", intro="", sets="") -> Line:
    return Line(text, asks, gives, role, needs, intro, sets)


# -- conversational moves ----------------------------------------------------

MOVES: dict[str, list[Line]] = {
    # Somebody puts something on the table. Openers are where the subject,
    # the culprit and the place get introduced, so most of them carry `intro`.
    "open": [
        L("{GUY} {DEED}.", gives="who", intro="guy thing"),
        L("SO I'M IN {PLACE} WITH {IT}.", gives="where", intro="place thing"),
        L("{IT!} IS GONE.", intro="thing"),
        L("I SAW {GUY} WITH {IT}.", gives="who", intro="guy thing"),
        L("YOU EVER NOTICE {IT!}?", intro="thing"),
        L("WHERE'S {IT!}.", asks="where", intro="thing"),
        L("DID YOU {DO} {IT}?", asks="why", intro="thing", sets="accuse"),
        L("I CAN'T {DO} {IT} ANY MORE.", intro="thing"),
        L("{IT!} CAME BACK.", intro="thing"),
        L("THEY'VE GOT {IT} IN {PLACE} NOW.", gives="where",
          intro="place thing"),
        L("I'VE BEEN THINKING ABOUT {IT}.", intro="thing"),
        L("{GUY} CALLED ABOUT {IT}.", gives="who", intro="guy thing"),
        L("SOMEBODY {DID} {IT}.", asks="who", intro="thing"),
        L("THERE WAS A SITUATION WITH {IT}.", intro="thing"),
        L("WE NEED TO TALK ABOUT {IT}.", intro="thing"),
        L("YOU {DID} {IT}. DIDN'T YOU.", intro="thing", sets="accuse"),
        L("SOMEBODY'S BEEN IN {PLACE} WITH {IT}.", gives="where",
          intro="place thing"),
    ],
    # The other one wants more before committing to anything.
    "probe": [
        L("WHERE.", asks="where"),
        L("WHEN.", asks="when"),
        L("WHO.", asks="who"),
        L("SO?", asks="why"),
        L("AND?", asks="why"),
        L("{IT}?"),
        L("WHAT ABOUT {IT}."),
        L("SINCE WHEN.", asks="when"),
        L("SAY THAT AGAIN."),
        L("IN {PLACE}?", asks="when"),
        L("WHY WOULD ANYONE {DO} {IT}.", asks="why"),
        L("HOW MUCH OF {IT}.", asks="many"),
        L("WHO TOLD YOU.", asks="who"),
        L("ON A WEEKDAY?", asks="when"),
        L("AND YOU'RE SURE IT WAS {GUY}.", asks="why"),
        L("WHAT WERE YOU DOING IN {PLACE}.", asks="why",
          role="A", needs="accused"),
        L("HOW MANY TIMES.", asks="many"),
        L("IS THAT A QUESTION."),
        L("AND NOBODY STOPPED YOU.", asks="who", role="A", needs="accused"),
        L("WAS ANYONE ELSE THERE.", asks="who"),
        L("YOU WENT BACK?", asks="when", role="A", needs="accused"),
        L("{ECHO}?"),
        L("{ECHO}. SINCE WHEN IS THAT A THING.", asks="when"),
    ],
    # More detail, which never helps. These are the answers.
    "restate": [
        L("I {DID} {IT}. THAT'S ALL.", gives="why", sets="admit"),
        L("{IT} WAS IN {PLACE}.", gives="where", intro="place"),
        L("{GUY} WAS THERE.", gives="who", intro="guy"),
        L("IT'S BEEN {N} DAYS.", gives="when"),
        L("I ONLY {DID} {IT} ONCE.", gives="many", sets="admit"),
        L("TWICE. MAYBE {N} TIMES.", gives="many", sets="admit"),
        L("IT WAS SITTING RIGHT THERE.", gives="where"),
        L("I HAD {IT} IN MY HAND.", sets="admit"),
        L("EVERYBODY SAW {IT}.", gives="who"),
        L("I WROTE {IT} DOWN."),
        L("IT WAS ALREADY OPEN WHEN I GOT THERE.", gives="why"),
        L("{GUY} SAID IT WAS FINE.", gives="who", intro="guy"),
        L("THERE WAS A LINE. I WAITED.", gives="why"),
        L("IT'S NOT LIKE I PLANNED IT.", gives="why", sets="admit"),
        L("I PUT {IT} BACK EXACTLY WHERE IT WAS.", gives="where",
          sets="admit"),
        L("NOBODY ELSE WANTED {IT}.", gives="why"),
        L("I DID WHAT ANYONE WOULD DO.", gives="why", sets="admit"),
        L("IT TOOK {N} MINUTES.", gives="when"),
        L("LAST TUESDAY. OR THE ONE BEFORE.", gives="when"),
        L("OUTSIDE {PLACE}. IN THE RAIN.", gives="where", intro="place"),
        L("BECAUSE OF {GUY}. OBVIOUSLY.", gives="why", intro="guy"),
        L("ALL OF IT. EVERY LAST BIT.", gives="many", sets="admit"),
        L("ABOUT {N} OF THEM.", gives="many"),
        L("A WEEK AGO. MAYBE TWO.", gives="when"),
        L("IN {PLACE}. WHERE ELSE.", gives="where", intro="place"),
        L("{GUY}. IT WAS {GUY}.", gives="who", intro="guy"),
    ],
    # Refusals. A dodge is tagged "~x": it acknowledges the question and
    # supplies nothing, which is a different thing from answering it.
    "deny": [
        L("I DIDN'T {DO} {IT}."),
        L("THAT'S NOT WHAT HAPPENED.", gives="~why"),
        L("I WASN'T ANYWHERE NEAR {PLACE}.", gives="~where"),
        L("NOBODY {DID} {IT}.", gives="~who"),
        L("I MAY HAVE MENTIONED {IT}. IN PASSING.", gives="~many",
          sets="admit"),
        L("THAT'S A DIFFERENT {NOUN}."),
        L("I WOULD REMEMBER {IT}.", gives="~when"),
        L("THAT WASN'T ME.", gives="~who"),
        L("YOU'RE THINKING OF SOMEBODY ELSE.", gives="~who"),
        L("I NEVER TOUCHED {IT}."),
        L("THAT'S A VERY STRONG WORD.", needs="accused"),
        L("I DON'T EVEN LIKE {IT}.", gives="why"),
        L("PROVE IT.", needs="accused"),
        L("NOT ONCE. NOT EVER.", gives="many"),
        L("I WAS IN {PLACE} ALL DAY.", gives="where", intro="place"),
        L("ASK {GUY} IF YOU DON'T BELIEVE ME.", gives="who", intro="guy"),
        L("{ECHO}. I NEVER SAID {ECHO}.", needs="echo:other"),
    ],
    # Pressing an accusation. Only the accuser gets these, and most of them
    # need the accusation to exist first.
    "escalate": [
        L("YOU {DID} {IT} ANYWAY.", role="A", sets="accuse"),
        L("IN FRONT OF {GUY}.", role="A", needs="accused"),
        L("NOTHING YOU DO IS IN PASSING.", role="A", needs="said:IN PASSING"),
        L("AND YOU SAID NOTHING.", role="A", needs="accused"),
        L("FOR {N} DAYS.", role="A", needs="accused", gives="when"),
        L("YOU DIDN'T EVEN ASK.", role="A", sets="accuse"),
        L("SO IT WAS YOU.", role="A", sets="accuse"),
        L("YOU'RE STANDING THERE AND YOU {DID} IT.", role="A", sets="accuse"),
        L("DON'T. I KNOW WHAT YOU DID.", role="A", sets="accuse"),
        L("I'M LOOKING RIGHT AT YOU.", role="A", sets="accuse"),
        L("AND YOU WEREN'T GOING TO SAY ANYTHING.", role="A", sets="accuse"),
        L("THIS IS THE PART WHERE YOU EXPLAIN.", role="A", sets="accuse"),
        L("WE HAD AN UNDERSTANDING ABOUT {IT}.", role="A", sets="accuse"),
        L("THAT'S THE SECOND TIME THIS MONTH.", role="A", needs="accused",
          gives="many"),
        L("AND {GUY} KNOWS.", role="A", needs="accused"),
        L("YOU DID IT IN {PLACE}. IN PUBLIC.", role="A", needs="accused",
          gives="where"),
        L("AND THEN YOU CAME HERE.", role="A", needs="accused"),
        L("THIS IS WHO YOU ARE NOW.", role="A", needs="denied"),
        L("I DEFENDED YOU.", role="A", needs="denied"),
        L("TO {GUY}. I DEFENDED YOU TO {GUY}.", role="A", needs="denied"),
        L("YOU'VE NEVER APOLOGISED FOR ANYTHING.", role="A", needs="denied"),
        L("AND NOW {IT} IS MY PROBLEM.", role="A", sets="accuse"),
        L("SAY IT OUT LOUD. SAY YOU {DID} IT.", role="A", needs="accused"),
        L("{ECHO}. LISTEN TO YOURSELF.", role="A", needs="echo:other"),
    ],
    # The format's favourite move: an entire person, settled in four words.
    "verdict": [
        L("HE'S A {TRAIT}.", needs="guy"),
        L("THAT'S {TRAIT} BEHAVIOUR."),
        L("SO HE'S A {TRAIT}. THAT'S THE WHOLE THING.", needs="guy"),
        L("YOU'RE DESCRIBING A {TRAIT}."),
        L("I'VE MET A {TRAIT} BEFORE.", needs="guy"),
        L("THAT IS TEXTBOOK {TRAIT}."),
        L("{GUY} IS A {TRAIT}. ALWAYS HAS BEEN.", intro="guy"),
        L("{ECHO}. THAT'S A {TRAIT} TALKING.", needs="echo:other"),
    ],
    "concede": [
        L("...MAYBE."),
        L("FINE."),
        L("I KNOW."),
        L("I KNOW IT'S NOT FINE."),
        L("FORGET {IT}."),
        L("I'M NOT GOING BACK."),
        L("LET'S NEVER {DO} {IT} AGAIN."),
        L("I'LL HANDLE {IT}."),
        L("YOU'RE RIGHT. I HATE THAT YOU'RE RIGHT.", role="D"),
        L("WE'RE NOT TALKING ABOUT THIS IN {PLACE}.", intro="place"),
        L("I'M LETTING IT GO. STARTING NOW.", role="A"),
        L("DO WHAT YOU WANT.", role="A"),
        L("OKAY. OKAY."),
    ],
    # The last line. It closes nothing.
    "button": [
        L("AND THAT'S WHY NOBODY GOES TO {PLACE}.", needs="place"),
        L("NOBODY SHOULD HAVE TO {DO} ANYTHING."),
        L("THAT'S THE SYSTEM WORKING."),
        L("SOMEBODY ALWAYS KNOWS."),
        L("NOBODY IS EVER ON THE COMMITTEE."),
        L("WE DON'T SPEAK OF {IT}."),
        L("I'M KEEPING {IT}."),
        L("THAT'S {N} DOLLARS I'M NOT GETTING BACK."),
        L("SO THAT'S WHERE WE ARE."),
        L("AND HE STILL HAS {IT}.", needs="guy"),
        L("{ECHO}. WRITE THAT DOWN.", needs="echo:other"),
    ],
    # Once in a while somebody says something true. Nobody is equipped for
    # it. The scene gives it exactly one deflection and then goes back to
    # being about a parking spot, which is the only way a line like this is
    # bearable in a sitcom.
    "existential": [
        L("DO YOU EVER THINK ABOUT HOW NONE OF THIS GETS REMEMBERED."),
        L("WE'RE ALL JUST WAITING IN DIFFERENT ROOMS."),
        L("SOMEBODY WILL SAY MY NAME FOR THE LAST TIME AND NOT KNOW IT."),
        L("I'VE BEEN HAVING THE SAME DAY FOR {N} YEARS."),
        L("NOTHING WE SAY IN HERE LEAVES THIS ROOM."),
        L("EVERYONE I'VE LOST STARTED OUT AS SOMEBODY I HADN'T MET."),
        L("THE BUILDING WILL OUTLIVE US AND IT ISN'T EVEN WELL BUILT."),
        L("WE KEEP TALKING SO WE DON'T FIND OUT WHAT'S UNDER IT."),
        L("SOMEBODY IS GOING TO HAVE TO THROW ALL MY THINGS AWAY."),
        L("I'M NOT AFRAID OF THE END. I'M AFRAID OF THE PART BEFORE."),
        L("{IT!} WILL STILL BE HERE WHEN I'M NOT.", intro="thing"),
        L("NONE OF THIS WAS EVER GOING TO MATTER AND WE DID IT ANYWAY."),
        L("I THINK I STOPPED BECOMING ANYTHING A WHILE AGO."),
        L("WE'RE ALL JUST PEOPLE WHO HAVEN'T BEEN FOUND OUT YET."),
        L("ONE OF US IS GOING TO BE THE LAST ONE IN THIS ROOM."),
        L("I DON'T REMEMBER AGREEING TO ANY OF THIS."),
    ],
    # The only available response to the above.
    "deflect": [
        L("ANYWAY."),
        L("SO ARE YOU GOING TO EAT THAT."),
        L("...WHAT."),
        L("DO YOU WANT A GLASS OF WATER."),
        L("THAT'S WHY I DON'T ASK YOU THINGS."),
        L("OKAY. WE'RE DOING THIS NOW."),
        L("IT'S TOO EARLY FOR THAT."),
        L("NOBODY IS TALKING ABOUT THAT."),
        L("AND ON THAT NOTE."),
    ],
    # The wildcard's register, when he is the one driving.
    "scheme": [
        L("I'VE GOT {IT!}. ALL OF IT.", gives="many", intro="thing"),
        L("WE BUY {IT!} WHOLESALE.", intro="thing"),
        L("NOBODY CHECKS {IT!}.", intro="thing"),
        L("GIVE ME {N} DAYS.", gives="when"),
        L("I KNOW A GUY IN {PLACE}.", gives="who", intro="place"),
        L("WE {DO} {IT!} BEFORE ANYONE NOTICES.", intro="thing"),
        L("THAT'S THE BEAUTY. THERE IS NO THEN.", needs="notfirst"),
        L("I'M TAKING {IT!} TO {PLACE}.", gives="where", intro="place thing"),
    ],
    # The neurotic's, when he is.
    "spiral": [
        L("I SAID {IT!} OUT LOUD.", intro="thing"),
        L("THEY WERE LOOKING AT ME."),
        L("DO YOU THINK PEOPLE NOTICE."),
        L("I'M BEING PHASED OUT."),
        L("I REHEARSED THE WHOLE THING."),
        L("IT'S BEEN {N} DAYS AND NOBODY HAS SAID ANYTHING.", gives="when"),
        L("SOMEBODY KNOWS ABOUT {IT!}.", intro="thing"),
        L("I CAN'T GO BACK TO {PLACE}.", gives="where", intro="place"),
    ],
    # The lead, alone at the mic. Second person, no reply coming.
    "bit": [
        L("WHAT IS THE DEAL WITH {IT!}?", intro="thing"),
        L("YOU EVER {DO} {IT!}? YOU DON'T {DO} IT. YOU SURRENDER TO IT.",
          intro="thing"),
        L("WHO DECIDED {IT!} WAS A WHOLE THING?", intro="thing"),
        L("THERE'S A GUY IN {PLACE} WHO KNOWS I TAKE {IT!}.",
          intro="place thing"),
        L("EVERYBODY HAS AN OPINION ABOUT {IT!}.", intro="thing"),
        L("SO I GO TO {PLACE} FOR {IT!}. BIG MISTAKE.", intro="place thing"),
        L("WHAT IS A MAN WHO {DID} {IT!}?", intro="thing"),
        L("{IT!}. NOBODY CAN DEFINE IT.", intro="thing"),
    ],
    "bit_tag": [
        L("NOBODY KNOWS. NOBODY HAS EVER KNOWN."),
        L("AND YET WE ALL AGREE TO IT."),
        L("SOME COMMITTEE. I WASN'T ON IT."),
        L("HUGE."),
        L("THAT'S THE SYSTEM WORKING."),
        L("I FIND THAT DEEPLY SUSPICIOUS."),
        L("AND WE JUST LET IT HAPPEN."),
    ],
}

# What can follow what. Nothing here describes a particular conversation --
# walking it produces one. The weights are the shape: an opener gets probed
# far more often than it gets agreed with, an escalation runs out into a
# verdict or a shrug.
FOLLOWS: dict[str, list[tuple[str, float]]] = {
    "open": [("probe", 6), ("restate", 3), ("escalate", 1)],
    "probe": [("restate", 6), ("deny", 3), ("concede", 1)],
    "restate": [("probe", 3), ("escalate", 3), ("verdict", 2), ("deny", 1),
                ("existential", 1.4)],
    "deny": [("escalate", 4), ("probe", 3), ("verdict", 2)],
    "escalate": [("deny", 3), ("concede", 3), ("verdict", 3), ("probe", 1),
                 ("existential", 1.6)],
    "verdict": [("concede", 3), ("probe", 2), ("button", 2)],
    "concede": [("button", 4), ("verdict", 1), ("existential", 2.0)],
    "button": [("concede", 1)],
    "existential": [("deflect", 10)],
    "deflect": [("probe", 3), ("button", 2), ("verdict", 1)],
    "scheme": [("probe", 5), ("deny", 2), ("escalate", 1), ("restate", 2)],
    "spiral": [("probe", 4), ("concede", 2), ("verdict", 1), ("restate", 2)],
}

# Which moves can end a scene. Reaching one is how an exchange stops, rather
# than by running out of turns in the middle of an argument.
TERMINAL = ("button", "verdict", "concede")

# How a mode opens, and how hot it runs.
MODES: dict[str, tuple[str, float]] = {
    "talk": ("open", 0.0),
    "dispute": ("open", 0.35),      # bias toward escalation
    "verdict": ("open", 0.2),
    "plan": ("open", 0.1),
    "standup": ("open", 0.0),
    "scheme": ("scheme", 0.1),
    "spiral": ("spiral", 0.1),
}


# -- the state of a conversation ---------------------------------------------


@dataclass
class Exchange:
    """What the room knows so far. Lines read from this and write to it."""

    scene: Scene
    turns: list[tuple[int, str]] = field(default_factory=list)
    accuser: int | None = None
    accused: int | None = None
    denied: bool = False
    admitted: bool = False
    # Who admitted it. Recorded but never read, `admitted` let the person
    # who had just said "it's not like I planned it" follow it with "you're
    # thinking of somebody else" -- the one thing in a conversation a
    # listener will not forgive.
    admitted_by: int | None = None
    # How many turns in a row have touched nothing already on the table.
    # Three is where a scene stops being about anything.
    drift: int = 0
    # (what was asked, who asked it). Only the person it was put to should
    # answer, and nothing may close the scene while it is open.
    open_q: tuple[str, int] | None = None
    patience: int = 0
    # (quotable tail, who said it). Throwing somebody's words back only works
    # if you know whose they were.
    echo: tuple[str, int] | None = None

    def transcript(self) -> str:
        return " ".join(t for _s, t in self.turns)


def last_clause(line: str) -> str:
    """The tail of a line, for the next speaker to throw back at it.

    "I MAY HAVE MENTIONED IT. IN PASSING." -> "IN PASSING". Quoting somebody
    is the cheapest way for two lines to be visibly about each other, and a
    short tail is a quote where a long one is a transcript.
    """
    for sep in (". ", ", "):
        if sep in line:
            line = line.rsplit(sep, 1)[1]
    tail = line.strip().rstrip(".?!")
    # Drop a leading subject, so the quote is the phrase and not the sentence.
    for lead in ("I NEVER ", "I DIDN'T ", "WHO SAYS I WAS ", "YOU'RE ",
                 "THAT'S ", "IT WAS ", "I WAS ", "NOBODY "):
        if tail.startswith(lead) and len(tail.split()) > 3:
            tail = tail[len(lead):]
            break
    words = tail.split()
    if not 1 <= len(words) <= 4:
        return ""
    # It has to be part of the line, not all of it.
    if len(words) >= len(line.strip().rstrip(".?!").split()):
        return ""
    return tail


def realise(skeleton: str, scene: Scene, rng: random.Random,
            other: str = "SOMEONE", echo: str = "") -> str:
    """Fill one skeleton against the scene.

    The subject is resolved lazily. Evaluating `scene.ref()` for a line that
    has no {IT} in it would still burn the "name it once" flag -- which is
    how a scene used to end up calling its own subject "it" without ever
    having said what it was.
    """
    out = skeleton
    if "{ECHO}" in out:
        out = out.replace("{ECHO}", echo or "THAT")
    if "{IT!}" in out:
        scene.said.add("thing")
        out = out.replace("{IT!}", scene.thing)
    if "{NOUN}" in out:
        out = out.replace("{NOUN}", scene.noun)
    if "{IT}" in out:
        out = out.replace("{IT}", scene.ref(rng))
    if "{GUY}" in out:
        scene.said.add("guy")
        out = out.replace("{GUY}", scene.guy)
    if "{PLACE}" in out:
        scene.said.add("place")
        out = out.replace("{PLACE}", scene.place)
    out = (
        out.replace("{DEED}", scene.deed)
        .replace("{DID}", scene.verb[1])
        .replace("{DO}", scene.verb[0])
        .replace("{TRAIT}", scene.trait)
        .replace("{N}", str(scene.number))
        .replace("{OTHER}", other)
    )
    return " ".join(out.split())


# -- legality ----------------------------------------------------------------


# Openings that point back at whatever was just said. "THAT'S CLOSE TALKER
# BEHAVIOUR" names nothing from the scene and still follows perfectly,
# because "that" is the previous line. Treating only the subject and the
# open question as connections undercounts these badly -- they are most of
# how people actually hold a thread.
_BACK_REFS = (
    "THAT", "SO ", "SO?", "AND ", "AND?", "YOU'RE ", "YOU ", "I'VE MET",
    "IS THAT", "WE'RE NOT", "DON'T", "OKAY", "FINE", "ANYWAY", "{ECHO}",
    "IT", "HE'S", "SHE'S", "NOBODY", "WHO SAYS",
)


def points_back(line: Line) -> bool:
    return line.text.startswith(_BACK_REFS)


def touches(line: Line, ex: Exchange) -> bool:
    """Does this line join on to anything already said?

    The subject, a quote of the last line, an answer to the question on the
    table, or a role in the argument. A line with none of them is a new
    sentence about a new thing, and three of those in a row is a scene that
    has wandered off the one it started.
    """
    if "{IT}" in line.text or "{IT!}" in line.text or "{ECHO}" in line.text:
        return True
    if ex.open_q is not None and line.gives.lstrip("~") == ex.open_q[0]:
        return True
    # A question is always about what was just said, whatever it names.
    if line.asks:
        return True
    if points_back(line):
        return True
    return line.role in ("A", "D") or bool(line.needs)


def introduces(line: Line, ex: Exchange) -> bool:
    """Would saying this put a person or a place on the table for the
    first time?"""
    for name, token in (("place", "{PLACE}"), ("guy", "{GUY}")):
        if token in line.text and name not in ex.scene.said:
            return True
    return False


def allowed(line: Line, ex: Exchange, slot: int, move: str = "") -> bool:
    """Could this character say this line, here, now?"""
    scene = ex.scene

    # Nobody denies what they have already admitted. The reverse is fine and
    # is the turn most of these scenes are built on.
    if move == "deny" and ex.admitted_by == slot and line.sets != "admit":
        return False

    # Role. Before anyone has been accused there are no roles to respect --
    # but a line written for a fight still presumes one, unless saying it is
    # what starts the fight.
    if ex.accused is not None:
        if line.role == "A" and slot != ex.accuser:
            return False
        if line.role == "D" and slot != ex.accused:
            return False
    elif line.role in ("A", "D") and line.sets != "accuse":
        return False

    need = line.needs
    if need == "accused" and ex.accused is None:
        return False
    if need == "denied" and not ex.denied:
        return False
    if need == "notfirst" and not ex.turns:
        return False
    if need == "guy" and "guy" not in scene.said:
        return False
    if need == "place" and "place" not in scene.said:
        return False
    if need.startswith("said:") and need[5:] not in ex.transcript():
        return False
    if need == "echo:self" and (ex.echo is None or ex.echo[1] != slot):
        return False
    if need == "echo:other" and (ex.echo is None or ex.echo[1] == slot):
        return False
    if "{ECHO}" in line.text and ex.echo is None:
        return False

    # Referents. A line may mention a place or a person only if it is allowed
    # to introduce one, or if the conversation has already introduced it.
    for name, token in (("place", "{PLACE}"), ("guy", "{GUY}")):
        if token in line.text and name not in line.intro \
                and name not in scene.said:
            return False
    return True


def _weighted(rng: random.Random, options: list[tuple[str, float]]) -> str:
    names = [n for n, _ in options]
    return rng.choices(names, weights=[w for _, w in options], k=1)[0]


def _can_answer(move: str, want: str) -> bool:
    return any(e.gives == want for e in MOVES.get(move, ()))


def _next_slot(ex: Exchange, move: str, prev: int, rng: random.Random) -> int:
    """Whose turn it is. Where there are roles, roles decide.

    An open question outranks a role: the person who asked it does not get to
    answer it, even if they are the accused and the move is a denial.
    """
    if ex.open_q is not None and move in ("restate", "deny"):
        asker = ex.open_q[1]
        return 1 - asker if asker in (0, 1) else rng.choice((0, 1))
    if ex.accused is not None:
        if move in ("deny", "concede", "restate"):
            return ex.accused
        if move in ("escalate", "verdict"):
            return ex.accuser
    # A third voice cuts in occasionally -- but never to answer for somebody
    # else, never twice running, and never while a question is open.
    if (prev != 2 and ex.open_q is None and ex.accused is None
            and move in ("probe", "verdict", "button", "existential")
            and rng.random() < 0.12):
        return 2
    return 1 - prev if prev in (0, 1) else rng.choice((0, 1))


def _pick(rng, move, ex, slot, used, avoid, late=False, insist=False):
    """The legal, preferably novel, preferably answering line for this turn."""
    legal = [e for e in MOVES[move] if allowed(e, ex, slot, move)]
    if not legal:
        return None
    # Answering the question on the table comes first and everything else
    # narrows within it. Applied the other way round, the continuity filters
    # could empty the pool of answers before anything had a chance to
    # prefer one, and the share of questions that got a direct answer fell
    # from 84% to 72%.
    if ex.open_q is not None:
        answers = [e for e in legal if e.gives == ex.open_q[0]]
        if answers:
            legal = answers
    if late:
        # Not in the last breath of a scene. Hearing about somebody's
        # brother-in-law for the first time as the closing line is the
        # written equivalent of a new character walking in during the credits.
        kept = [e for e in legal if not introduces(e, ex)]
        if kept:
            legal = kept
    if insist:
        # The turn before touched nothing already on the table. This one has
        # to come back to it or the scene stops being about anything.
        kept = [e for e in legal if touches(e, ex)]
        if kept:
            legal = kept
    # Say what this is about. A scene that gets two turns in without naming
    # its own subject is a scene where "it" has no antecedent, and every
    # line after that is about nothing.
    if "thing" not in ex.scene.said and len(ex.turns) >= 1:
        names = [e for e in legal if "{IT!}" in e.text or "{IT}" in e.text]
        if names:
            legal = names
    fresh = [e for e in legal if e.text not in used and e.text not in avoid]
    pool = fresh or [e for e in legal if e.text not in used] or legal
    return rng.choice(pool)


def generate(rng: random.Random, scene: Scene, mode: str = "talk",
             min_turns: int = 4, max_turns: int = 7,
             avoid: set[str] | None = None,
             trace: list | None = None) -> list[tuple[int, str]]:
    """Walk the move machine against a growing record of the conversation.

    Returns (speaker slot, line) pairs. Slot 0 opens and slot 1 answers; the
    director decides who those are. Slot 2 cuts in occasionally.

    Selection order per turn: which move may legally follow, then whose turn
    it is under that move, then which of that move's lines this person could
    actually say given what has been established, then whether it answers the
    open question, and last whether we have used it recently.
    """
    opener, heat = MODES.get(mode, MODES["talk"])
    avoid = avoid if avoid is not None else set()
    ex = Exchange(scene=scene)
    move = opener
    spoken = opener          # the last move actually said, not the next one
    slot = 0
    used: set[str] = set()

    for _ in range(max_turns):
        # Near the end nothing new gets introduced, and after two turns that
        # touched nothing the next one has to come back to the subject.
        late = len(ex.turns) >= min_turns - 2
        insist = ex.drift >= 1
        line = _pick(rng, move, ex, slot, used, avoid, late,
                     insist or slot == 2)
        if line is None:
            # Nothing in this move is sayable by this person right now --
            # which is information, not a failure. Fall back to a move that
            # always has something legal in it.
            for fallback in ("probe", "restate", "concede"):
                line = _pick(rng, fallback, ex, slot, used, avoid, late,
                             insist or slot == 2)
                if line is not None:
                    move = fallback
                    break
        if line is None:
            break

        used.add(line.text)
        avoid.add(line.text)
        spoken = move
        text = realise(line.text, scene, rng,
                       echo=ex.echo[0] if ex.echo else "")
        ex.turns.append((slot, text))
        if trace is not None:
            trace.append((move, slot, line, text, ex.open_q, ex.accused))

        # -- write back what this line established --------------------------
        ex.drift = 0 if touches(line, ex) else ex.drift + 1
        if (line.sets == "accuse" or move == "escalate") and ex.accused is None:
            ex.accuser = slot
            ex.accused = 1 - slot if slot in (0, 1) else 0
        if line.sets == "admit":
            ex.admitted = True
            ex.admitted_by = slot
        if move == "deny":
            ex.denied = True
        if line.asks:
            ex.open_q, ex.patience = (line.asks, slot), 0
        elif ex.open_q is not None:
            if line.gives == ex.open_q[0]:
                ex.open_q = None
            else:
                # A dodge buys exactly one turn of pressing. After that the
                # question is gone, which people do, but it must not look
                # like the machine simply forgot.
                ex.patience += 1
                if ex.patience > 1 or not line.gives.startswith("~"):
                    ex.open_q = None
        tail = last_clause(text)
        ex.echo = (tail, slot) if tail else None

        # -- where next -----------------------------------------------------
        if move in TERMINAL and ex.open_q is None and len(ex.turns) >= min_turns:
            break
        options = list(FOLLOWS.get(move, FOLLOWS["restate"]))
        if heat and move != "escalate":
            options.append(("escalate", heat * 6))
        # A denial needs something to deny. Without an accusation, a question
        # or a challenge in front of it, it is a man arguing with the room.
        if (ex.accused is None and ex.open_q is None
                and move not in ("probe", "escalate")):
            options = [(m, w) for m, w in options if m != "deny"] or options
        if len(ex.turns) < 3:
            options = [(m, w) for m, w in options
                       if m != "existential"] or options
        if ex.admitted_by is not None:
            options = [(m, w) for m, w in options
                       if m != "deny"
                       or _next_slot(ex, m, slot, rng) != ex.admitted_by
                       ] or options
        # Nobody gets a third turn running, and nobody follows their own
        # question at all -- barring only the third let somebody ask "what
        # about it." and then answer it themselves on the very next line.
        held = None
        if len(ex.turns) >= 2 and ex.turns[-1][0] == ex.turns[-2][0]:
            held = ex.turns[-1][0]
        elif line.asks or move == "probe":
            held = slot
        if held is not None:
            options = [(m, w) for m, w in options
                       if _next_slot(ex, m, slot, rng) != held] or options
        if ex.open_q is not None:
            answering = [(m, w) for m, w in options
                         if _can_answer(m, ex.open_q[0])]
            options = answering or options
            options = [(m, w) for m, w in options if m not in TERMINAL] or options
        # The nearer the end, the heavier the ending.
        pressure = (len(ex.turns) / max_turns) ** 2
        options = [(m, w * (1 + 4 * pressure)) if m in TERMINAL else (m, w)
                   for m, w in options]
        if len(ex.turns) + 1 < min_turns:
            options = [(m, w) for m, w in options if m not in TERMINAL] or options
        move = _weighted(rng, options)
        slot = _next_slot(ex, move, slot, rng)
        # `_next_slot` draws its own randomness, so filtering the options by
        # what it *would* return and then calling it again was two different
        # draws: the floor went back to the same person anyway. Enforce it
        # on the answer rather than on the question.
        if held is not None and slot == held:
            slot = rng.choice([x for x in (0, 1) if x != held])

    # A scene ends; it does not stop. If the loop ran out mid-argument, land
    # it -- an abandoned accusation is the thing that reads as broken.
    if ex.turns and spoken not in TERMINAL:
        # An open question is answered before anything is allowed to close.
        closers = ["concede", "button", "verdict"]
        if ex.open_q is not None:
            closers.insert(0, "restate")
        held = (ex.turns[-1][0]
                if len(ex.turns) >= 2 and ex.turns[-1][0] == ex.turns[-2][0]
                else None)
        for closer in closers:
            slot = _next_slot(ex, closer, ex.turns[-1][0], rng)
            if slot == held:
                slot = rng.choice([s for s in (0, 1) if s != held])
            line = _pick(rng, closer, ex, slot, used, avoid, late=True)
            if line is None:
                continue
            avoid.add(line.text)
            text = realise(line.text, scene, rng,
                           echo=ex.echo[0] if ex.echo else "")
            ex.turns.append((slot, text))
            if line.gives == (ex.open_q[0] if ex.open_q else None):
                ex.open_q = None
            if trace is not None:
                trace.append((closer, slot, line, text, ex.open_q, ex.accused))
            if closer in TERMINAL:
                break
            # Answering was not an ending; land it properly.
            for final in ("concede", "button"):
                prev = slot
                slot = _next_slot(ex, final, slot, rng)
                if slot == prev and len(ex.turns) >= 2                         and ex.turns[-2][0] == prev:
                    slot = rng.choice([s for s in (0, 1) if s != prev])
                line = _pick(rng, final, ex, slot, used, avoid, late=True)
                if line is not None:
                    avoid.add(line.text)
                    text = realise(line.text, scene, rng,
                                   echo=ex.echo[0] if ex.echo else "")
                    ex.turns.append((slot, text))
                    if trace is not None:
                        trace.append((final, slot, line, text, ex.open_q,
                                      ex.accused))
                    break
            break
    return ex.turns


def standup(rng: random.Random, scene: Scene,
            avoid: set[str] | None = None) -> list[tuple[int, str]]:
    """A set, not a conversation: premise then one or two tags, one voice."""
    avoid = avoid if avoid is not None else set()
    lines = []
    for pool, count in ((MOVES["bit"], 1), (MOVES["bit_tag"], rng.randint(1, 2))):
        # Sampled without replacement: the same tag twice in a row is the
        # one thing a comedian never does.
        fresh = [e for e in pool if e.text not in avoid] or pool
        for line in rng.sample(fresh, min(count, len(fresh))):
            avoid.add(line.text)
            lines.append((0, realise(line.text, scene, rng)))
    return lines


# -- the other registers -----------------------------------------------------

# Dropped into the middle of an exchange. Nobody acknowledges it; that is the
# whole trick. Generated against the same scene, so it is wrong about
# something the conversation was actually discussing.
SLIPS = [
    "{IT!} IS IN THE WALLS.",
    "I HAVEN'T SLEPT SINCE {N}.",
    "THERE ARE {N} OF ME.",
    "DON'T SAY THAT NEAR THE WINDOW.",
    "I USED TO LIVE IN {PLACE}. INSIDE IT.",
    "THE CEILING KNOWS.",
    "THEY REPLACED {OTHER} AND NOBODY SAID ANYTHING.",
    "{IT!}. {IT!}. {IT!}.",
    "I CAN HEAR THE BUILDING.",
    "WHAT YEAR IS THIS.",
    "{GUY} HAS BEEN DEAD FOR {N} YEARS.",
    "I'M NOT SUPPOSED TO BE IN THIS ROOM.",
    "{PLACE} ISN'T THERE ANY MORE.",
    "I {DID} {IT!} IN A DREAM FIRST.",
]

# Kept short so they fit on one line even at the smallest base resolution --
# the strip does not wrap, and a clipped laugh is worse than no laugh.
LAUGHS = [
    "[ LAUGHTER ]", "[ LAUGHTER ]", "[ LAUGHTER ]", "[ LAUGHTER ]",
    "[ BIG LAUGH ]", "[ BIG LAUGH ]",
    "[ SCATTERED LAUGHS ]",
    "[ APPLAUSE ]",
    "[ HELD TOO LONG ]",
    "[ A BEAT TOO LATE ]",
    "[ ONE PERSON LAUGHS ]",
    "[ AUDIENCE UNSURE ]",
]

# Captions that are not laughter -- the room itself.
STAGE_DIRECTIONS = [
    "[ A DOOR CLOSES SOMEWHERE ]",
    "[ NOBODY MOVES ]",
    "[ THIS GOES ON FOR SOME TIME ]",
    "[ EVERYONE LOOKS AT {NAME} ]",
    "[ A PHONE RINGS ONCE ]",
    "[ SOMEONE IS ON THE FIRE ESCAPE ]",
    "[ {NAME} IS STILL HOLDING {IT!} ]",
]

# When the broadcast itself breaks.
GLITCH = [
    "SCENE {N}. SAME APARTMENT.",
    "PLEASE STAND BY",
    "LAUGH TRACK NOT RESPONDING",
    "CONTINUITY ERROR -- {IT!}",
    "[ THIS BIT IS MISSING ]",
    "RESUMING",
    "{NAME} IS NOT IN THIS SCENE",
]

COLD_OPEN = [
    "[ WE JOIN THIS ALREADY IN PROGRESS ]",
    "[ PREVIOUSLY: NOTHING ]",
    "[ LIVE FROM AN APARTMENT THAT DOES NOT EXIST ]",
]

# Everyone says it. Nobody explains why.
RITUAL = [
    "{IT!}.",
    "{IT!}. {IT!}.",
    "AGAIN.",
    "SAY IT.",
]

# Picked once near startup, realised once, then reused verbatim forever.
CALLBACK_SETUP = [
    "NOT THAT THERE'S ANYTHING WRONG WITH {IT!}",
    "AND THAT'S {IT!} FOR YOU",
    "{IT!}. WHO'S WITH ME.",
    "A CLASSIC {TRAIT} MOVE",
    # No adjective in front of the slot. This line wanted "A BIG" and the
    # subject supplies its own words, so it collided both ways: {IT!} brings
    # a determiner ("A BIG THE SECOND DATE") and {NOUN} collides with the
    # nouns that are already adjectival -- "A BIG BIG SALAD", which is a
    # deliberate reference and so not going anywhere. {IT!} alone cannot
    # collide with either.
    "THAT'S {IT!}, IF TRUE",
]
