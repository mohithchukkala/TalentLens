# Related work and research foundation

**Purpose.** To place the project in the literature honestly: what it builds on, what it only read about, and where it differs.
**What this is not.** It is not a systematic literature review (about 15 sources were checked), so it makes **no claim of
novelty and no claim of a research gap**. Where the project's design was motivated by something, that is stated as a design
motivation, and "informed" is used only where the repository records the decision or a measurement supports it.

## How to read this document
| Tag | Meaning |
|---|---|
| **[P]** | published research or a textbook |
| **[D]** | official software documentation |
| **[M]** | model documentation (the model card) |
| **[O]** | this project's own observation, measured on its synthetic evaluation set (not literature) |

**Verification, stated plainly.** Every source below was found through web search and its bibliographic details were confirmed
against a publisher, proceedings, arXiv, ACL Anthology or official documentation page (links in the reference list). I did **not** read
most papers in full: statements about papers rest on abstracts and listing pages, except the DPR paper, whose text I extracted from
the PDF to read the hybrid-retrieval section and Table 2. Numbers quoted from papers are what those pages report; I did not reproduce them.
**None of the cited methods was implemented in this project**, except where "used" is written (TF-IDF, a published sentence-embedding model,
scikit-learn and PyMuPDF functionality).

---

## 1. Keyword-based matching and term weighting

**1.1 Spärck Jones (1972), "A statistical interpretation of term specificity and its application in retrieval" [P].**
*Problem:* how much should matching a term count? *Technique:* weight terms by how rare they are in the collection, so that matches on
rarer terms count for more; the paper reports considerable improvements on three test collections with this simple procedure.
*Relevance:* the origin of inverse document frequency, one half of TF-IDF (attribution to this paper is made by the publisher's abstract
and a secondary page; I did not read the full text). *Difference:* in this project the IDF is fitted on the job description and **one**
resume (pairwise), i.e. two documents, so IDF is almost flat: shared terms get 1.0 and unique terms 1.405 [O, by hand calculation checked in a test].
Spärck Jones's idea needs a real collection to matter; the pairwise choice was made for batch-independence (a resume scored 0.2553 in a batch of 10 and
0.2134 alone [O]), not for retrieval quality. *Informed:* the idea that rare shared terms deserve more weight; the pairwise fitting is this project's own trade-off.

**1.2 Salton and Buckley (1988), "Term-weighting approaches in automatic text retrieval" [P].**
*Problem:* which term-weighting schemes work best for automatic retrieval. *Technique:* a study of weighting systems; its abstract describes
single-term indexing with well-chosen weights as a strong approach and offers baseline single-term models against which more elaborate analysis can be compared.
*Relevance:* the usual citation for TF-IDF-style vector-space matching, and the justification for treating TF-IDF plus cosine as the traditional baseline (Method A).
*Difference:* the exact variant used here (sublinear `1 + ln(tf)`, smoothed IDF, L2 normalisation, 1-2 grams) is scikit-learn's, documented in 5.1, not
taken from this paper. *Informed:* the decision to include TF-IDF as the baseline against which the semantic and combined methods are compared (the repository records this decision from Phase 1; the paper supports its reasonableness).

**1.3 Manning, Raghavan and Schütze, *Introduction to Information Retrieval* (Cambridge University Press, 2008) [P].**
*Problem/technique:* a textbook on retrieval: vector-space scoring, cosine similarity and ranked-retrieval evaluation. I confirmed the book,
its authors and its free online edition; I did not check chapter contents against this project's statements, so it is cited as general background for cosine similarity, TF-IDF and Precision/Recall/MRR, not for a specific formula.

**1.4 Keyword matching in real applicant-tracking systems.** I did not find a primary (peer-reviewed or official) source describing how commercial
systems match keywords, so this document **does not describe them**. The project's keyword baseline is TF-IDF, a standard information-retrieval method, and its measured behaviour is in section 4.

**1.5 Raghavan, Barocas, Kleinberg and Levy (2020), "Mitigating bias in algorithmic hiring: evaluating claims and practices" (FAT\* 2020) [P].**
*Problem:* what vendors of algorithmic pre-employment assessments disclose about building and validating their tools and about bias. *Technique:* analysis of
vendors' public materials from technical and legal angles (per the abstract). *Relevance:* context for caution about claims made by hiring algorithms and for human oversight.
*Difference:* it studies assessment vendors, not resume ranking or keyword matching, and does not evaluate this kind of system. *Informed:* nothing in the code; it is consistent with, and cited for, the project's statements that scores are not verified competence and that nobody is rejected automatically.

## 2. Transformer-based semantic similarity

**2.1 Devlin et al. (2019), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (NAACL-HLT 2019) [P].**
Pretrains deep bidirectional representations from unlabelled text; background to everything below. Not used directly: the project never loads BERT itself.

**2.2 Reimers and Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (EMNLP-IJCNLP 2019) [P].**
*Problem:* BERT needs both sentences passed through the network together, which makes similarity search over many sentences very slow. *Technique:* siamese and triplet
networks that produce a fixed-size embedding per sentence, comparable with cosine similarity; the paper reports reducing the time to find the most similar pair among 10,000 sentences from about 65 hours to about 5 seconds.
*Relevance:* the approach behind the sentence-transformers library and models. *Difference:* this project uses a published model from that library as-is; it implements no siamese training and no fine-tuning.
*Informed:* the structure of Method B: embed each text independently once and compare with cosine similarity (the repository's design; the paper explains why this scales). The decision to use cosine on unit-length vectors is the library's.

**2.3 Wang et al. (2020), "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers" (NeurIPS 2020) [P].**
*Problem:* compress large pretrained transformers. *Technique:* distil the self-attention of the teacher's last layer into a smaller student; the authors report retaining more than 99% of accuracy on SQuAD 2.0 and several GLUE tasks with 50% of the teacher's parameters and computation (from the listing and abstract).
*Relevance:* the model card says the sentence model was fine-tuned from `nreimers/MiniLM-L6-H384-uncased`, a MiniLM-family base. *Difference:* the paper is about general-purpose compression, not sentence embeddings or recruiting; I did not verify that the base model was produced exactly as in the paper. *Informed:* nothing directly; the choice of a small CPU-friendly model was this project's own criterion.

**2.4 Model card, `sentence-transformers/all-MiniLM-L6-v2` [M] (official Hugging Face page, read in Phase 12).** It states: maps sentences and paragraphs to a **384-dimensional** dense vector; input text longer than **256 word pieces is truncated**
by default; base model `nreimers/MiniLM-L6-H384-uncased`; contrastive fine-tuning on more than 1 billion sentence pairs (its dataset table totals 1,170,060,424 tuples; 100k steps, batch size 1024); the training sequence length was limited to **128 tokens**;
license **Apache-2.0**; intended as "a sentence and short paragraph encoder" for retrieval, clustering and sentence similarity; no dedicated limitations section.
*Relevance and informed decisions:* (a) the 256-word-piece limit is why the project chunks long documents; the same limit was measured locally (`max_seq_length` 256, 384 dimensions) and a test shows truncation really discards the tail [O]. (b) "short paragraph encoder" and the 128-token training length mean resume-length text is
outside the model's intended range, which supports chunking; however the chunk size (200 tokens, fixed in Phase 2 below the 254 usable tokens) is larger than the 128-token training length, and this project did not evaluate the effect of that choice [O: not evaluated, unchanged]. (c) The licence permits use in a coursework project.
*Limitations:* the card says nothing about recruiting, fairness or negation; the project's own evaluation shows the semantic score ignores negation and "describes the role from outside" text [O].

**2.5 Sentence-Transformers documentation, "Pretrained Models" [D].** Recommends `all-mpnet-base-v2` for best quality and describes `all-MiniLM-L6-v2` as about five times faster while still offering good quality; it also stresses that experimentation is key and that leaderboard results may not transfer to your own task.
*Informed:* the trade-off between quality and CPU speed behind choosing the small model. *Limitation:* the project did not compare models; a larger model might rank differently [O: not evaluated].

## 3. Hybrid and combined ranking

**3.1 Karpukhin et al. (2020), "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020) [P].**
*Problem:* first-stage passage retrieval for question answering, where sparse methods such as TF-IDF and BM25 were standard. *Technique:* a dual-encoder dense retriever; the paper also tests a **hybrid** that reranks the union of BM25 and dense top-2000 candidates with the linear score
`BM25(q, p) + λ · sim(q, p)`, with `λ = 1.1` chosen on the development set (read from the PDF text). Its Table 2 (top-20 accuracy, single DPR vs BM25+DPR) shows the hybrid helping on some datasets and not on others, for example 63.2 to 71.5 (SQuAD) and 79.8 to 85.2 (TREC) but 78.4 to 76.6 (Natural Questions) and 73.2 to 71.0 (WebQuestions) *(values read from extracted PDF text; check against the PDF before quoting)*.
*Relevance:* a published precedent for adding a lexical score to an embedding score linearly, and a reminder that the sum of two differently scaled scores needs a weight. *Differences:* it is question answering, not resume ranking; the weight is tuned on development data, whereas this project fixed 0.50 / 0.20 / 0.30 in advance and never tuned them; this project's lexical term is TF-IDF (not BM25) and it adds a skill-coverage term.
*Informed:* the idea of a weighted sum of lexical and semantic scores. The paper's finding that a hybrid is not uniformly better is consistent with, but does not explain, this project's result that the combined index was **not clearly better than semantic alone** [O].

**3.2 Thakur et al. (2021), "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" (NeurIPS 2021, Datasets and Benchmarks) [P].**
*Problem:* how retrieval models generalise zero-shot across domains. *Technique:* a benchmark of 18 datasets evaluating 10 systems (lexical, sparse, dense, late-interaction, re-ranking); it reports BM25 as a robust baseline, re-ranking and late-interaction models best on average at higher computational cost, and dense and sparse models efficient but often underperforming.
*Relevance:* supports keeping a lexical baseline next to a semantic model instead of assuming the neural method wins. *Difference:* its lexical baseline is BM25, this project's is TF-IDF; its domains and scale are very different. *Informed:* consistent with the project's design of comparing methods on identical inputs; the repository's reasons for doing so are its own.

**3.3 Gugnani and Misra (2020), "Implicit skills extraction using document embedding and its use in job recommendation" (AAAI-20 / IAAI-20) [P].**
*Problem:* matching resumes to job descriptions when many skills are implied rather than written. *Technique:* NLP skill extraction (the paper reports precision 0.78 and recall 0.88 on an industrial dataset) plus **implicit** skills inferred from similar job descriptions in a Doc2Vec space trained on 1.1 million job descriptions; it reports a mean reciprocal rank of 0.88, a 29.4% improvement over a baseline using only explicit skills.
*Relevance:* the closest published precedent I verified for combining explicit skill extraction with embedding-based similarity in resume/job matching, and evidence that explicit skills alone miss relevant information. *Differences:* this project extracts **explicit** skills only, from a hand-written dictionary, infers none, uses a sentence-embedding model rather than Doc2Vec, and is evaluated on 46 synthetic resumes, not industrial data. *Informed:* nothing in the code; it supports the stated limitation that a dictionary cannot see implicit or paraphrased skills (observed in the project's paraphrase resumes [O]).

**3.4 Zhu et al. (2018), "Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning" (ACM Transactions on Management Information Systems; arXiv 1810.04040) [P].**
*Problem:* quantitative person-job fit. *Technique:* a convolutional neural network (PJFNN) learning joint representations of resumes and job postings from historical applications. *Relevance:* a supervised alternative. *Differences:* it needs a large history of hiring outcomes, which this project does not have and which would be sensitive personal data; this project trains nothing. The arXiv preprint was read, the final journal version was not.
*Informed:* none; the decision not to train a supervised model came from the lack of labelled real data.

**3.5 What I did not find.** No verified source describing this project's exact combination (TF-IDF cosine + sentence-embedding cosine + dictionary skill coverage with fixed weights). That absence in a small search is not evidence that no such work exists, and no novelty is claimed. (Two other candidate papers on resume matching with all-MiniLM-L6-v2 and NER turned up in search, but I could not open them and therefore do not cite them.)

## 4. Evaluation, fairness and responsible use

**4.1 Järvelin and Kekäläinen (2002), "Cumulated gain-based evaluation of IR techniques" (ACM TOIS 20(4)) [P].** Proposes cumulative-gain measures that use graded relevance, the basis of nDCG, and discount gain at lower ranks. *Used:* nDCG@3 with graded labels 0-3. *Difference:* the gain `2^label - 1` and `log2(position + 1)` discount are a common variant; I did not verify that the original paper uses exactly these, so the project documents its own definition (`EXPERIMENT_RESULTS.md`).

**4.2 Bolukbasi et al. (2016), "Man is to computer programmer as woman is to homemaker? Debiasing word embeddings" (NeurIPS 2016) [P].** Shows that word embeddings trained on news text encode gender stereotypes and proposes a debiasing method. *Relevance:* a reason to expect that embedding-based scores may carry social bias. *Differences:* it studies word embeddings, not sentence-transformer models, and this project performed **no** bias or fairness evaluation; names and other personal details stay in the text the embedding model sees (only e-mail addresses, phone numbers and URLs are removed). *Informed:* the stated limitation and the human-review requirement, not any measured result.

**4.3 European Commission AI Act Service Desk, page "Employment" (Annex III point 4(a) of Regulation (EU) 2024/1689) [D].** The page says AI systems used for recruitment or selection may be considered high-risk, and gives as examples tools that rank or score candidates where the ranking serves as a primary input for decision-making even though recruiters can override it; some tools such as CV organisation and interview scheduling can fall outside high-risk status.
*Relevance:* context for why a ranking tool like this one needs human oversight, documentation and care if it were ever used for real. *Limits:* this is not legal advice; the project is an unreleased student project, has not been placed on any market, and makes **no compliance claim**. I did not open the regulation text itself, only the Commission page; an application date mentioned in one secondary source was not verified and is not used.

## 5. Software documentation relied on
- **scikit-learn `TfidfTransformer` documentation [D]** (the page read is for version 1.9.1, the version used here): smoothed IDF `idf(t) = log[(1 + n) / (1 + df(t))] + 1` and `sublinear_tf` (`tf -> 1 + log(tf)`). *Informed:* the TF-IDF implementation; the formulas were also checked numerically against `TfidfVectorizer` in the tests [O].
- **PyMuPDF documentation, "Text" recipes [D]:** notes that text may not come out in reading order and describes `sort` and text blocks with position information to establish a reading order. *Informed:* the parser reads blocks sorted from top-left. *Limitation:* that page does not cover scanned pages or OCR; this project has no OCR, and multi-column layouts remain untested on real resumes [O].

---

## 6. Design motivations (not a claimed research gap)
These are the reasons the system is built the way it is. Each is supported by the project's own measurements [O], not by a literature gap.

1. **Exact skill coverage vs lexical similarity.** Coverage asks a yes/no question per required skill (is "PostgreSQL" present, under any listed alias?) and returns a fraction a recruiter can read. Lexical similarity (TF-IDF) returns a weighted overlap of all words, so a long resume with many generic words can score like one with the key skills. Coverage is more interpretable but only as good as the dictionary: it missed paraphrased experience (coverage 0 to 0.2 for resumes labelled relevant) and counted mentions regardless of use (coverage 1.0 for a resume stating it had "never used any of these") [O].
2. **TF-IDF captures overlap, not equivalence.** "sklearn" and "scikit-learn" are different tokens; "trained neural networks" shares no word with "machine learning engineer" and scored 0.0 although a semantic model scored it above an unrelated resume (0.307 vs 0.138 in a small synthetic check) [O]. Conversely keyword lists ranked above strong candidates in three of five jobs [O].
3. **Semantic similarity relates meanings but does not establish possession of a skill.** The embedding score reflected topic: resumes that were negations or described the role from outside (recruiter, designer, manager, technical writer) scored 0.53 to 0.75, in the range of relevant partial matches, and the model has no notion of "has not used" [O].
4. **Why compare several methods.** Each fails on different inputs; a single number hides this. Comparing methods on the same inputs and labels is also the only way to say anything honest about whether a combination helps. In this project the combination beat TF-IDF but did not clearly beat the semantic score alone, and per job it won on two and lost on three on nDCG [O]. The literature on hybrids (DPR, BEIR) likewise reports that benefits vary by dataset.
5. **Interpretable components and CSV export.** The final index is an ordering aid with fixed, visible weights; showing the raw components, the matched and missing skills and the evidence lets a person check any ranking, notice when one term dominates (skill coverage accounts for about 75% of the combined index's variance on the evaluation set [O]) and keep the records outside the app. This follows from the project's human-oversight stance, not from a gap in the literature.

## 7. Contribution, stated carefully
What the repository **does** contain: an integrated, reproducible, locally runnable implementation that puts three scoring methods side by side on identical inputs; a written evaluation protocol and a frozen synthetic labelled set with an honest account of what it can and cannot show; robust PDF handling; tests that include a real-model integration layer; a security review that found and fixed a real display vulnerability; and documentation that separates measured results from assumptions.
What it **does not** show: that the combined index outperforms semantic similarity alone, that any method works on real resumes, that skill coverage is a reliable measure of competence, or that any part of the design is new.

---

## Reference list
1. K. Spärck Jones, "A statistical interpretation of term specificity and its application in retrieval", *Journal of Documentation* 28(1), 11-21, 1972. DOI: [10.1108/eb026526](https://doi.org/10.1108/eb026526)
2. G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval", *Information Processing & Management* 24(5), 513-523, 1988. DOI: [10.1016/0306-4573(88)90021-0](https://doi.org/10.1016/0306-4573(88)90021-0)
3. C. D. Manning, P. Raghavan and H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008. Online edition: <https://nlp.stanford.edu/IR-book/>
4. M. Raghavan, S. Barocas, J. Kleinberg and K. Levy, "Mitigating Bias in Algorithmic Hiring: Evaluating Claims and Practices", ACM FAT\* 2020. arXiv: <https://arxiv.org/abs/1906.09208>
5. J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL-HLT 2019, 4171-4186. DOI 10.18653/v1/N19-1423. <https://aclanthology.org/N19-1423/>
6. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP-IJCNLP 2019. DOI 10.18653/v1/D19-1410. <https://aclanthology.org/D19-1410> (arXiv: <https://arxiv.org/abs/1908.10084>)
7. W. Wang, F. Wei, L. Dong, H. Bao, N. Yang and M. Zhou, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers", *Advances in Neural Information Processing Systems 33* (NeurIPS 2020). <https://papers.nips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html> (arXiv: <https://arxiv.org/abs/2002.10957>)
8. Model card: `sentence-transformers/all-MiniLM-L6-v2`, Hugging Face. <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
9. Sentence Transformers documentation, "Pretrained Models". <https://www.sbert.net/docs/sentence_transformer/pretrained_models.html>
10. V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", EMNLP 2020, 6769-6781. arXiv: <https://arxiv.org/abs/2004.04906>
11. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava and I. Gurevych, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", NeurIPS 2021 Datasets and Benchmarks. arXiv: <https://arxiv.org/abs/2104.08663>
12. A. Gugnani and H. Misra, "Implicit Skills Extraction Using Document Embedding and Its Use in Job Recommendation", *Proceedings of the AAAI Conference on Artificial Intelligence* 34(8), 13286-13293, 2020. DOI 10.1609/aaai.v34i08.7038. <https://ojs.aaai.org/index.php/AAAI/article/view/7038>
13. C. Zhu, H. Zhu, H. Xiong et al., "Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning", *ACM Transactions on Management Information Systems*, 2018 (as listed on arXiv). arXiv: <https://arxiv.org/abs/1810.04040>
14. K. Järvelin and J. Kekäläinen, "Cumulated gain-based evaluation of IR techniques", *ACM Transactions on Information Systems* 20(4), 422-446, 2002. DOI: [10.1145/582415.582418](https://doi.org/10.1145/582415.582418)
15. T. Bolukbasi, K.-W. Chang, J. Zou, V. Saligrama and A. Kalai, "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings", NeurIPS 2016. <https://proceedings.neurips.cc/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html> (arXiv: <https://arxiv.org/abs/1607.06520>)
16. European Commission, AI Act Service Desk, "Employment" (Annex III point 4(a), Regulation (EU) 2024/1689). <https://ai-act-service-desk.ec.europa.eu/en/node/464>
17. scikit-learn documentation, `TfidfTransformer` (version 1.9.1). <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html>
18. PyMuPDF documentation, "Text" recipes. <https://pymupdf.readthedocs.io/en/latest/recipes-text.html>

**Link status:** items 1, 2, 4-7, 10-15 were confirmed through search listings (publisher, proceedings, ACL Anthology, arXiv); items 8, 9, 16-18 pages were opened and read; item 3 was confirmed through search listings. The DOI links for 1, 2 and 14 were taken from those listings and not individually resolved by me. Item 10's PDF text was read directly.
