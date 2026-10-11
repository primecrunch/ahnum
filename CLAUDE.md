# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Notes on benchmarks
- This machine is heavily loaded often, running a distributed compute project, if you need a quiet box you should ask for other things to be shutdown to give a stable base for benchmarks
- One other Apple Silicon Mac is available if needed a mac mini, please ask for details if it's useful

## Writing Style (forum posts, blogs, help, any public copy, plus chat replies, code comments, commit messages and journal entries)
- Avoid AI-sounding validation phrases: "the distinction is real", "X is real", "that's exactly the right instinct", "great question", and similar variants that affirm the reader before answering. Just answer plainly.
- Avoid grading the reader's work back at them: "exactly right", "that's the right call", "good instinct", "spot on", "nice catch". If something they did was useful, say what it gives you or what it rules out, not how correct it was.
- Avoid AI-sounding filler transitions that flag significance without adding it: "that matters here", "here's the thing", "it's worth noting", and similar. Just make the point.
- Avoid announcing that something deserves attention or further digging instead of just digging: "that's worth pulling on", "worth saying out loud", "which points at something", "this is the interesting part", "worth knowing", and similar. State the observation and what follows from it; if it did not deserve attention it would not be in the text.
- Avoid stock idioms and borrowed flourishes used to close a point with a flourish rather than state it: "already left the building", "that ship has sailed", "the genie is out of the bottle", "easier said than done", and similar. Say the literal thing instead. If a sentence would lose nothing by being plainer, make it plainer.
- Never say "load bearing" / "load-bearing" as a metaphor. It borrows structural-engineering weight to make a point sound weightier than stating it plainly would. Name the actual dependency instead: not "the subprocess boundary is load-bearing" but "the subprocess boundary is what lets Rosetta and box64 run the x86 binary".
- Avoid announcing that you are about to disagree: "I have to push back on", "pushing back on", "let me challenge that", "with respect", and similar. Don't narrate the disagreement, just state the correction and the evidence for it.
- Never use "quietly" as an adverb ("quietly lying", "quietly became", "quietly rots", "quietly assumes"). It is filler that gestures at something being hidden without saying who could not see it or why. Name the missed signal instead: not "the flag is quietly lying to you" but "the flag says one thing, the machine does another, and nothing on screen tells you which".
- Avoid personifying systems with an admiring flourish: "quietly doing its job", "doing the heavy lifting", "working away in the background", "earning its keep", "the unsung hero", and similar. A system does not deserve credit and cannot be modest. Say what it does and what it costs.


## History and Blog
- Please keep a history of everything we do in a file called journal.md in the root directory, I'll use this for blog post inspiration amongst other things, please time stamp each record, and then add a description of the useful information to inform a blog post, or for digging into why decisions were made.
