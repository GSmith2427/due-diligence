# ADR-0001: Leadership data sources

- **Status**: Accepted
- **Date**: 2026-05-27
- **Deciders**: Project author
- **Tags**: data-sources, legal, ethics

## Context

The system needs accurate, current information on the leadership of public companies — names, titles, tenure, prior roles, board memberships, education, compensation, and any flagged conflicts of interest. This data is the foundation of the "leadership effectiveness" analysis that motivates the project, so it needs to be reliable, attributable to a source, and obtainable without ongoing manual intervention.

The obvious candidate — and the one originally proposed in the project brief — is LinkedIn. It has the broadest coverage of executive profiles, employment histories, and educational backgrounds of any single source on the web. Scraping LinkedIn would also be the most direct path: one source, one parser, comprehensive results.

There are three problems with that path.

**Terms of service.** LinkedIn's User Agreement explicitly prohibits scraping, automated data collection, and using bots to access the platform. While the *hiQ Labs v. LinkedIn* decisions established that scraping publicly accessible data is not a Computer Fraud and Abuse Act violation, that ruling does not override LinkedIn's contractual restrictions, and LinkedIn has continued to pursue civil action against scrapers. For a public portfolio project, advertising a LinkedIn scraper carries reputational risk independent of the legal question.

**Technical fragility.** LinkedIn aggressively fingerprints, rate-limits, and blocks scrapers. Maintaining a working scraper requires constant adversarial engineering against an actively-hostile target. This work has no portfolio value and produces a system that breaks frequently in production.

**Data quality.** LinkedIn profiles are self-reported and unverified. Executives misrepresent dates, inflate titles, and omit unflattering roles. For a due-diligence system, primary-source verification is the whole point.

## Decision

The system will obtain executive and leadership data primarily from **SEC EDGAR filings**, supplemented by **company-published leadership pages**, **Wikipedia/Wikidata**, and **news article extraction**. LinkedIn will not be scraped.

The data sources, in order of precedence:

1. **SEC DEF 14A proxy statements.** Public companies are legally required to disclose, for each named executive officer and director: full name, age, position, tenure, prior roles for the past five years, education, board memberships at other public companies, compensation, and material conflicts of interest. This is the gold standard.
2. **SEC 10-K Item 10.** Annual report section covering "Directors, Executive Officers and Corporate Governance," used to cross-reference and fill gaps in the proxy statement.
3. **Company leadership/about pages.** Scraped where ToS permits (most corporate sites permit indexing for non-commercial research). Useful for current titles and bios written by the company itself.
4. **Wikipedia and Wikidata APIs.** For longer-tenured executives, Wikipedia often has biographical context (early career, notable past employers, public controversies) that doesn't appear in SEC filings. Wikidata provides this in structured form.
5. **News article extraction.** For recent appointments and material events not yet reflected in annual filings. Sourced via a news search API (Tavily or Brave), with content extracted on demand.

Every data point will carry a `provenance` field recording the source URL, retrieval timestamp, and a content hash. The final report will cite primary sources directly so the user can verify any claim.

## Consequences

### Positive

- **Defensible.** The system relies on legally-mandated public disclosures and standard web APIs. No ToS violations, no legal grey zones.
- **Higher data quality.** SEC filings are verified, attested, and subject to enforcement. Self-reported LinkedIn data is not.
- **Stable.** SEC EDGAR has provided a stable public API for decades and offers free bulk-download archives. No adversarial maintenance.
- **Auditable.** Every claim in a generated report traces back to a primary source the user can read.
- **Demonstrates judgment.** Choosing the harder-but-better path is itself a signal of engineering maturity.

### Negative

- **Coverage limited to US public companies.** SEC filings only cover companies that file with the SEC. Private companies, foreign issuers without ADRs, and pre-IPO startups will have thin or no data. The system's scope is bounded accordingly.
- **Latency.** Proxy statements are annual; a new CEO appointed in March won't appear in an SEC filing until the next proxy season. The news-extraction source partially mitigates this but introduces its own noise.
- **Parsing complexity.** SEC filings are XBRL-tagged HTML with inconsistent structure across companies. Extraction is meaningfully harder than scraping a structured social profile would be. This work is absorbed into the leadership-collection agent.
- **No social-graph data.** LinkedIn's connection data (who an executive knows, who they previously worked with) is not available from any of the chosen sources. The system therefore cannot model professional networks. If a future version needs this, it would have to come from a paid data provider (BoardEx, Equilar) or a research-licensed dataset, not from scraping.

## Alternatives considered

- **LinkedIn scraping.** Rejected for the reasons given in Context.
- **Paid data providers (BoardEx, Equilar, S&P Capital IQ).** Rejected: violates the project's free/self-hosted constraint and would gate the entire system behind a vendor relationship.
- **Hugging Face datasets / academic dumps.** Rejected as a *primary* source because they are point-in-time snapshots and go stale. May be used as an offline test fixture for evaluation.
- **Google Knowledge Graph API.** Rejected: deprecated for new applications, replaced by paid Google Enterprise Knowledge Graph.

## Open questions

- How to handle executives appointed between annual filings — is news extraction sufficient or does an 8-K monitor need its own agent?
- Whether to add a manual-override interface so users can supply corrections when SEC data is stale.

## References

- *hiQ Labs, Inc. v. LinkedIn Corp.*, 31 F.4th 1180 (9th Cir. 2022) — CFAA does not prohibit scraping publicly accessible data, but the case was ultimately settled and ToS-based claims were preserved.
- SEC EDGAR documentation: <https://www.sec.gov/edgar/sec-api-documentation>
- LinkedIn User Agreement § 8.2: prohibition on automated data collection.
