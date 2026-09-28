# Remittance research — Korea → Nepal

Working notes toward a graduate application in **ICTD / Information Systems**, studying
digital remittance from inside the industry.

**The question:** *Why do Nepali workers in Korea keep using informal channels when cheaper,
legal, faster digital ones exist?*

Four candidate explanations, all testable:

| | |
|---|---|
| **KYC / documentation** | Workers with precarious visa status fail identity checks. AML rules built for financial crime exclude the poorest senders. |
| **Recipient-side access** | The app's quality is irrelevant if the receiving household is hours from a cash-out point. |
| **Trust and social ties** | The informal operator is someone's cousin. The app is a foreign company. |
| **Exchange-rate opacity** | "Zero fee" hides the spread. True cost = fee + spread. Senders may be reading it correctly. |

---

## Reading list

PDFs are **not** committed — see `.gitignore`. Download them yourself from the links below.

### Minjin Kim — Yonsei (Barun ICT Research Center)
PhD, University of East Anglia 2023. Works on whether FinTech actually delivers financial
inclusion. **Her position is sceptical** — note the word "myth" in the thesis title.

| Read | Where | Access |
|---|---|---|
| **PhD thesis** — *The Myth of Financial Inclusion through FinTech: the Digital Credit Industry in Kenya* | [UEA repository](https://ueaeprints.uea.ac.uk/id/eprint/92118/) | free |
| **Digital credit for all?** *Information Technology for Development*, 2024 | [author PDF](https://ueaeprints.uea.ac.uk/id/eprint/96735/) · [doi](https://doi.org/10.1080/02681102.2024.2402996) | free |
| **Fintech for the poor?** *Development Studies Research*, 2025 | [doi](https://doi.org/10.1080/21665095.2025.2547852) | open access |

**The finding to quote:** mobile banking loans are *less* accessible to women, less-educated
people and **casual workers**, while both loan types reach rural users equally.
Migrant workers are casual workers.

### Wenlong Bian — Sungkyunkwan (Associate Professor of Finance, CEPR)
How culture and social structure shape FinTech use.

| Read | Where | Access |
|---|---|---|
| **Clan culture and participation in FinTech-based risk sharing**, 2024 | [doi](https://doi.org/10.1016/j.pacfin.2024.102259) | paywalled — request from author |
| **Uncovering the Mystery of China's Success in FinTech**, 2026 | [doi](https://onlinelibrary.wiley.com/doi/10.1111/acfi.70233) | paywalled |

### Yong Suk Kim — Sungkyunkwan (MIS) · `yongskim@skku.edu`
Publishes as **Yongsuk Kim**. Social ties and platform behaviour.

| Read | Where | Access |
|---|---|---|
| **External Bridging and Internal Bonding**, *MIS Quarterly* 2018 | [AIS eLibrary](https://aisel.aisnet.org/misq/vol42/iss1/15/) | often free |
| **Crowdfunding from friends: tie strength and embeddedness**, *Decision Support Systems* 2023 | search the exact title on Google Scholar | paywalled |

---

## How to read these (≈35 min each)

1. Abstract — what they found
2. Last two paragraphs of the introduction — their claimed contribution
3. Conclusion and Discussion — what it means
4. Figures and the main results table
5. **Limitations / Future research** — read this closely

Skip the literature review, the econometric specification and the robustness checks.
You are not replicating the statistics.

**The Limitations section is the highest-value paragraph in any of these papers.** It is where
the author states in print what they could not do — the data they lacked, the population they
could not reach. If your corridor or your access solves one of their stated limitations, that
is your opening sentence.

## Requesting a paywalled paper

Emailing an author for a copy is normal and they almost always send it. Authors may legally
share their own work. It also makes a good low-pressure first contact: ask for the paper, say
briefly why, and you have a thread to continue once you have read it.

---

## What a developer contributes

Not econometrics. Instruments.

1. **Onboarding-funnel analysis.** Every user who abandons KYC is a data point on who digital
   finance excludes — with the exact screen where it happened. Economists infer exclusion from
   surveys; frontend telemetry observes it.
2. **A remittance price observatory.** Query every KRW→NPR provider daily; compute the true
   cost as rupees delivered per 100,000 won, fee *and* spread. The World Bank does this
   manually and quarterly for major corridors. A continuous dataset for this corridor does not
   exist. See `pricing/`.
3. **A field instrument.** A Nepali-language survey distributed through migrant networks —
   primary data no Korean academic can collect, and unlike company data it belongs to you.

## Data access — settle this early

Company transaction data needs employer permission and is governed by Korea's PIPA. Assume the
workable version is **aggregated and anonymised**. Never promise a supervisor data you have not
confirmed you can provide. Keep a fallback (surveys, interviews, public pricing) that needs no
company data at all.

## Notes

Per-paper notes live in `notes/`. One file per paper: what they argue, the finding that matters,
their stated limitations, and what it connects to in the corridor.
