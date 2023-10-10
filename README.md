# Overlap

Checks overlap between inputs or outputs from a test set (e.g. MATH), and a corpus (e.g. open-web-math).

Example:
```bash
python check_overlap.py --test-dataset MATH \
  --test-key input \
  --dataset open-web-math/open-web-math \
  --ngram-n 30
```
This command checks whether 30-grams from `MATH` `input` sequences appear in `open-web-math`.

See `notebooks/analysis.ipynb` for an example usage of the output.

### Model generations (Llemma lm-evaluation-harness)
```bash
python check_overlap.py --test-dataset /path/to/output.json \
  --test-key input \
  --dataset open-web-math/open-web-math \
  --ngram-n 30
```
Where `output.json` is produced by the [Llemma `lm-evaluation-harness`](https://github.com/wellecks/lm-evaluation-harness). \
The JSON file must have a sequence stored at a `unprocessed_answers` key in the `metadata`. \
The `minerva_math_xyz` tasks yield JSON that adheres to this format.

See `notebooks/analysis.ipynb` for an example usage of the output.

### Authors:
- Sean Welleck, Keiran Paster

### Llemma
This tool was developed as part of the Llemma project.
Llemma's analysis is saved in the `llemma` branch.
### Citation:
Please cite the following:
```
@article{azerbayev2023llemma,
    title={Llemma: an open language model for mathematics},
    author={Zhangir Azerbayev and Hailey Schoelkopf and Keiran Paster and Marco Dos Santos and Stephen McAleer and Albert Q. Jiang and Jia Deng and Stella Biderman and Sean Welleck},
    eprint={xyz.xyz},
    archivePrefix={arXiv}
    year={2023}
}
```
