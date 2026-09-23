# CalibMI — paper project

One result set, many renderings. `main_tmlr.tex` is the source of truth;
everything else is a projection of it. Never edit a number in a section file
unless the generating script produced it.

## Main files

| File | Target | Status |
|---|---|---|
| `main_tmlr.tex` | **TMLR** — full manuscript, primary target | ~Oct 27 |
| `main_embc.tex` | EMBC 2027, 4 pages IEEEtran | Jan 24, 2027 — currently 5pp, needs trimming |
| `main_bigdata.tex` | 5-page IEEE rendering. **Not submitted to BigData 2026.** | Retarget BigData 2027 |

Set the main document in Overleaf (menu top-left) to whichever you are
compiling. TMLR is double-blind — before submitting, swap the `article`
preamble for TMLR's official style and anonymise.

## Shared sections

`sections/` is `\input` by every main file.

- `intro_full.tex`, `discussion_full.tex` — **rewrite banners inside; these are
  raw material, not final prose**
- `related_work.tex`, `protocol.tex`, `design.tex` — done
- `results_n9.tex` — done
- `results_n54.tex` — **empty until P1 and P2 run**
- `limitations.tex` — done

## Abstracts

Written fresh per audience; every number quoted must appear in `main_tmlr.tex`.
Each file carries its deadline, audience notes, and blockers.

| File | Venue | Due |
|---|---|---|
| `abstracts/ieee_brain.tex` | IEEE Brain Discovery & Neurotech | Sep 30 priority / Oct 19 |
| `abstracts/stanford_aihealth.tex` | Stanford AI+HEALTH | Oct 15 |
| `abstracts/aan_abstract.tex` + `aan_research_report.tex` | AAN Neuroscience Research Prize | Oct 20 |
| `abstracts/bci_meeting.tex` | International BCI Meeting 2027 | Nov 9 – Jan 15 |
| `abstracts/_paper_abstract.tex` | the manuscript's own abstract | — |

## Build

```bash
latexmk -pdf main_tmlr.tex
```

Compile locally rather than depending on Overleaf's server, and keep the repo
as the source of truth so the freeze in
`docs/unified_project_and_freeze.md` §4.3 tags what was actually submitted.
