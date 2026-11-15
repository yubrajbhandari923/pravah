# Reference audit (E19)

**Date:** 2026-11-14.
**Method:** every entry of `reports/quarto/references.bib` was checked with the research-desk reference-lookup tool
(`verify_bibtex`, OpenAlex and arXiv). The check matches by DOI where one exists, otherwise by title, and compares the
title, first author and year. No reference was added or invented. The citation keys used in the `.qmd` files resolve
in the bib file (`scripts/validate_report.py`).

**Result:** 20 / 20 entries verified.

| key | matched by | status | note |
|---|---|---|---|
| green2006 | title | verified | OpenAlex W2584579713 lists the year as 2007; the bib keeps 2006 (thesis dated July 2006), noted in the entry |
| pennington2024 | DOI 10.5194/acp-24-2345-2024 | verified | |
| schlutow2026 | DOI 10.5194/gmd-19-6259-2026 | verified | |
| mons2017 | DOI 10.1016/j.jweia.2017.07.007 | verified | |
| bhardwaj2026 | title / arXiv 2608.00050 | verified | preprint, not peer reviewed |
| manohar2018 | DOI 10.1109/MCS.2018.2810460 | verified | |
| zhou2022 | DOI 10.1021/acsenvironau.1c00051 | verified | |
| mano2022 | DOI 10.3390/s22103808 | verified | |
| krause2008 | title | verified | no DOI in the bib (OpenAlex gives 10.5555/1390681.1390689) |
| choi2025 | DOI 10.1088/1748-9326/ae2b89 | verified | plan-03's literature line calls it `choi2026`; the record's year is 2025 |
| bartos2021 | DOI 10.1029/2020WR029551 | verified | |
| roy2018 | DOI 10.3390/s18061778 | verified | |
| zheng2009 | DOI 10.1016/j.atmosenv.2009.01.014 | verified | |
| sharma2019 | DOI 10.1016/j.buildenv.2019.05.026 | verified | |
| cheng2022 | DOI 10.1016/j.automatica.2021.110112 | verified | |
| webster2022 | DOI 10.1029/2022JD036469 | verified | |
| mues2018 | DOI 10.5194/gmd-11-2067-2018 | verified | |
| raanes2024 | DOI 10.21105/joss.05150 | verified | |
| desilva2021 | DOI 10.21105/joss.02828 | verified | |
| karnik2026 | DOI 10.21105/joss.09265 | verified | |

## Scope of the check

"Verified" means that the record exists and its metadata match. It does **not** mean that each citing sentence is
supported by the cited work. That was not checked here; `docs/research/literature-map.md` holds the reading notes.

## References cited in text but not in the bib

- **ERA5** (Hersbach et al., 2023, doi:10.24381/cds.adbb2d47) and **WorldPop** (Nepal 2020, doi:10.5258/SOTON/WP00671)
  appear as attribution text in chapter 17, not as bib entries.
- **Unchecked:** these dataset DOIs were not run through the lookup tool (dataset DOIs are not OpenAlex works). The E17
  journal notes that the WorldPop landing page content was not checked.
- Recommendation: add them as `@misc` entries before a manuscript, after confirming them by hand.
- `patel2026` (a GitHub repository with no bibliographic record) is deliberately excluded, per the bib header.
