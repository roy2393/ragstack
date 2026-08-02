# 📊 CUAD Legal RAG Diagnostics & Evaluation Report
**Run ID:** `test_cuad_run_recursive_mock_hybrid`
**Date:** 2026-08-02 11:35:36

---

# Abstract
This diagnostic report profiles the CUAD dataset and evaluates retrieval boundaries, chunk coverage, and LLM reasoning behaviors to trace legal-domain RAG compliance and failure modes.

---

# 1. Dataset Understanding & Profiling
This section profiles the legal clauses, question types, and contract dimensions in the Atticus Contract Understanding dataset.

## A. Dataset Statistics

| Metric | Value |
| --- | --- |
| **Total Evaluation Samples** | 510 |
| **Unique Contracts Profiled** | 102 |
| **Average Words per Contract** | 6250.3 |
| **Average Characters per Contract** | 42585.8 |
| **Longest Contract (Words)** | 33,948 |
| **Smallest Contract (Words)** | 123 |
| **Average Answer Length (Words)** | 118.0 |

## B. Clause category distribution
Counts and ASCII bar chart representing the occurrence frequency of legal clause categories.

| Clause Category | Count | Distribution Bar |
| --- | ---: | :--- |
| Notice Period To Terminate Renewal | 17 | `███████████████` |
| Affiliate License-Licensee | 17 | `███████████████` |
| Warranty Duration | 16 | `██████████████░` |
| Cap On Liability | 16 | `██████████████░` |
| No-Solicit Of Employees | 16 | `██████████████░` |
| License Grant | 16 | `██████████████░` |
| Liquidated Damages | 16 | `██████████████░` |
| Anti-Assignment | 15 | `█████████████░░` |
| No-Solicit Of Customers | 15 | `█████████████░░` |
| Rofr/Rofo/Rofn | 15 | `█████████████░░` |
| Renewal Term | 15 | `█████████████░░` |
| Volume Restriction | 15 | `█████████████░░` |
| Governing Law | 15 | `█████████████░░` |
| Competitive Restriction Exception | 15 | `█████████████░░` |
| Non-Disparagement | 14 | `████████████░░░` |
| Exclusivity | 14 | `████████████░░░` |
| Minimum Commitment | 14 | `████████████░░░` |
| Ip Ownership Assignment | 13 | `███████████░░░░` |
| Most Favored Nation | 13 | `███████████░░░░` |
| Expiration Date | 13 | `███████████░░░░` |
| Uncapped Liability | 13 | `███████████░░░░` |
| Joint Ip Ownership | 13 | `███████████░░░░` |
| Affiliate License-Licensor | 11 | `█████████░░░░░░` |
| Effective Date | 11 | `█████████░░░░░░` |
| Post-Termination Services | 11 | `█████████░░░░░░` |
| Revenue/Profit Sharing | 11 | `█████████░░░░░░` |
| Change Of Control | 11 | `█████████░░░░░░` |
| Source Code Escrow | 11 | `█████████░░░░░░` |
| Document Name | 11 | `█████████░░░░░░` |
| Unlimited/All-You-Can-Eat-License | 11 | `█████████░░░░░░` |
| Third Party Beneficiary | 10 | `████████░░░░░░░` |
| Irrevocable Or Perpetual License | 10 | `████████░░░░░░░` |
| Audit Rights | 10 | `████████░░░░░░░` |
| Covenant Not To Sue | 10 | `████████░░░░░░░` |
| Non-Transferable License | 10 | `████████░░░░░░░` |
| Termination For Convenience | 9 | `███████░░░░░░░░` |
| Agreement Date | 9 | `███████░░░░░░░░` |
| Non-Compete | 8 | `███████░░░░░░░░` |
| Parties | 8 | `███████░░░░░░░░` |
| Insurance | 7 | `██████░░░░░░░░░` |
| Price Restrictions | 5 | `████░░░░░░░░░░░` |

## C. Question Types Clustered

| Question Type | Count | Distribution Bar |
| --- | ---: | :--- |
| Yes/No Clause Presence | 199 | `███████████████` |
| General Clause Extraction | 113 | `████████░░░░░░░` |
| Date/Duration/Timeline | 105 | `███████░░░░░░░░` |
| Party/Entity Identification | 45 | `███░░░░░░░░░░░░` |
| Governing Law/Jurisdiction | 26 | `█░░░░░░░░░░░░░░` |
| Financial/Monetary Obligation | 22 | `█░░░░░░░░░░░░░░` |

---

# 2. Retrieval & Chunking Understanding
We trace the efficiency of context retrieval at a sentence-level grain, verifying whether the actual ground-truth utilized clauses were successfully retrieved inside the prompt context.

## A. Retrieval Success Metrics

| Metric | Value | Explanation |
| --- | ---: | :--- |
| **Recall@k Success Rate** | **0.00%** | Percentage of queries where at least one utilized sentence was retrieved |
| **Clause Chunk Coverage** | **0.00%** | Average percentage of all utilized sentences successfully retrieved |

## B. Chunk Boundary Analysis
Measures whether recursive/semantic chunking split legal clauses across boundaries, which often degrades retriever similarity scores.

| Boundary Category | Percentage | Count |
| --- | ---: | ---: |
| Clause fully contained in one chunk | 0.00% | 0 |
| Clause split across multiple chunks | 0.00% | 0 |
| Clause not retrieved | 100.00% | 1 |

---

# 3. LLM Behavior & Performance Breakdown

| Metric | Value |
| --- | ---: |
| **Average Answer Length (Words)** | 6.0 words |
| **Average Pipeline Latency** | 5.0 ms |
| **Latency Standard Deviation** | 0.0 ms |

---

# 4. Failure Analysis Dashboard
This section aggregates pipeline run failures by classifying them into distinct root causes.

| Failure Cause | Count | Distribution Bar |
| --- | ---: | :--- |
| Poor Retrieval (Relevance < 0.4) | 0 | `░░░░░░░░░░░░░░░` |
| LLM Hallucination (Relevance >= 0.5, Adherence < 0.6) | 2 | `███████████████` |
| Incomplete Answer (Completeness < 0.6) | 0 | `░░░░░░░░░░░░░░░` |
| Model Refusal (Unsupported label when info present) | 0 | `░░░░░░░░░░░░░░░` |
| Poor Chunking (Sentence split across boundaries) | 0 | `░░░░░░░░░░░░░░░` |
| Other failures | 0 | `░░░░░░░░░░░░░░░` |

---

# 5. Qualitative Case Studies (Sample Failures)
Below are detailed examples of queries that failed evaluation parameters.

### Case Study 1: LLM Hallucination (Relevance >= 0.5, Adherence < 0.6)
- **Question:** Is there a fee increase or consent requirement, etc. if one party’s use of the product/services exceeds certain threshold?
- **Ground Truth Answer:** *Yes, based on the context provided in the software license and maintenance agreement:

Section 2 - Grant of License states that Sparkling's right to use the Software and Documentation at locations other than the Authorized Locations is subject to the payment of additional fees in accordance with the Fee Schedule.

This indicates that if Sparkling wants to use the licensed software and documentation at locations beyond the Authorized Locations listed in Schedule C, they would need to pay additional fees as outlined in the Fee Schedule.

So there is a requirement for Sparkling to pay additional fees if they want to expand their use of the licensed software and documentation beyond the initially authorized locations.*
- **Generated Response:** `MOCK RESPONSE: placeholder for pipeline testing.`
- **Metrics:** Adherence: `0.50`, Relevance: `0.50`, Completeness: `0.50`

### Case Study 2: LLM Hallucination (Relevance >= 0.5, Adherence < 0.6)
- **Question:** Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?
- **Ground Truth Answer:** *No, there is no clause in the Consulting Agreement that grants a right of first refusal, right of first offer, or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products, or services. The agreement does not contain any such provisions.*
- **Generated Response:** `MOCK RESPONSE: placeholder for pipeline testing.`
- **Metrics:** Adherence: `0.50`, Relevance: `0.50`, Completeness: `0.50`


*Last updated: 2026-08-02 11:35:36*