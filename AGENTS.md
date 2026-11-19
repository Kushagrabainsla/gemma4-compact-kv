# AGENTS.md

Context file for AI agents working in this repository.

## This repository is generated
- A build script in the author's private research repository creates this repository:
  `thesis/code-release/build_release.py` in the `masters-thesis` repository. On the author's computer, that
  repository is `../masters-thesis/`, next to this folder.
- Do not edit files here by hand. A rebuild replaces every file except `.git`. Change the source in the research
  repository, run the build script, check the diff, then commit and push here with the author's OK.
- The research journal (`RESEARCH_JOURNAL.md`), the review write-ups (`dossier/`) and the experiment protocol
  (`PROTOCOL.md`) that the notes name are in the private repository. They are not public.

## This repository is public
- Never add private notes, credentials, account records (Pod ids, cost ledgers, Pod registries) or the token ids of
  private text. The build script removes them; `README.md` ("What is not in this release") lists what it removes.
- Before a push, run `thesis/code-release/scan_release.py` in the research repository.
- Do not change code files that a run manifest pins by SHA-256: the tests check them.

## Style
- Write in ASD-STE100 (Simplified Technical English): one topic in each sentence, active voice, simple tenses, no
  em dashes.
