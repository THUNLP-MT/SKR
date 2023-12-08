# Overview
The repo contains the codes for **[Self-Knowledge Guided Retrieval Augmentation for Large Language Models]**(https://aclanthology.org/2023.findings-emnlp.691.pdf) (EMNLP Findings 2023)
![Method_overview](figs/skr.png)

## Data
The Temporal dataset we use is in the fold `data/`.
- `Question`: The question.
- `Gold answer`: The answer.
- `passages`: The retrieved passages from wikipedia.

## Chain-of-Thought Results
- The CoT and retrieval-augmented CoT results are given in the fold `results/`, where the `chain_of_thought_gpt3` indicates the responses.

## Steps
- For SKR_prompt and SKR_icl, we use the prompts shown in the paper to elicit the self-knowledge of the dev data directly.

- For SKR_cls, we use the training data and train a [BERT classifier](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) to elicit the self-knowledge of the dev data. We use the settings with `lr=2e-5` and `epochs=10`.

- For SKR_knn, the steps are as follows:
    - cd `source/` , collect the self-knowledge of the training data, run `skr.py` and get the `train_skr.json` file.
    - run `knn.py` to use the self-knowledge to the dev data and get the `dev_skr_knn.json` file.
    - run `eval_skr.py` to evaluate the results.

## Citation

```bibtex
@inproceedings{wang-etal-2023-self-knowledge,
    title = "Self-Knowledge Guided Retrieval Augmentation for Large Language Models",
    author = "Wang, Yile  and
      Li, Peng  and
      Sun, Maosong  and
      Liu, Yang",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.691",
    pages = "10303--10315",
}
```

## Acknowledgement
- [Rethinking with Retrieval](https://github.com/HornHehhf/RR)
