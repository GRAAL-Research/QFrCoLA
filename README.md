# QFrCoLA: a Quebec-French Corpus of Linguistic Acceptability Judgments

This repository contains the official QFrCoLA: a Quebec-French Corpus of Linguistic Acceptability Judgments dataset (see
dataset directory) and the source code used to generate the tables and training of the model illustrated
in [QFrCoLA: a Quebec-French Corpus of Linguistic Acceptability Judgments]().

## About the Dataset

<img width="254" height="59" alt="image" src="https://github.com/user-attachments/assets/838a7219-f0a6-4a6c-a649-4785a3d2de95" />

Available on [HuggingFace](https://huggingface.co/datasets/graalul/qfrcola).

### Example of Sentences in the Dataset

![img.png](figs/img.png)


### Statistics About the Dataset Compared to Other Similar Datasets
> Statistics are divided by splits.

![img_1.png](figs/img_1.png)

### Dataset Structure

#### Data Fields

- `label`: the binary label of the sentence, where `0` means ungrammatical and `1` means grammatical.
- `sentence`: the sentence.
- `source`: the URL source of the sentence.
- `category`: the aggregated BDL category of the sentence linguistic phenomena.

### Download the Dataset

You can manually download our dataset splits available in `dataset`, or you can use the HuggingFace dataset class as
follows:

```python
from datasets import load_dataset

dataset = load_dataset("davebulaval/qfrcola")
```

### License

This dataset is under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

### To Cite

```
@inproceedings{beauchemin2025qfrcola,
  title={QFrCoLA: a Quebec-French Corpus of Linguistic Acceptability Judgments},
  author={Beauchemin, David and Khoury, Richard},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={119--130},
  year={2025}
}
```

## About the Source Code

In the directory `article_src`, you can find the source code used to clean the dataset and compute the statistics and
in `la_tda`, the code is used to fine-tune all our models. The code was adapted from the official repository of the
article [Acceptability Judgements via Examining the Topology of Attention Maps](https://arxiv.org/pdf/2205.09630v2.pdf).

## Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.

<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">QFrCoLA: a Quebec-French Corpus of Linguistic Acceptability Judgments</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">QFrCoLA</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/GRAAL-Research/qfrcola</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">QFrCoLA is a dataset of binary normative linguistic acceptability judgments in Quebec French, with in-domain sentences from the Banque de dépannage linguistique (BDL) and out-of-domain sentences from the Académie française.
    </code></td>
  </tr>
    <tr>
        <td>creator</td>
        <td>
          <div itemscope itemtype="http://schema.org/person" itemprop="creator">
            <table>
              <tr>
                <th>property</th>
                <th>value</th>
              </tr>
                <tr>
                <td>name</td>
                <td><code itemprop="name">David Beauchemin</code></td>
              </tr>
              <tr>
                <td>sameAs</td>
                <td><code itemprop="sameAs">https://scholar.google.com/citations?hl=fr&user=ntoPgSUAAAAJ</code></td>
              </tr>
              </tr>
              <tr>
                <td>name</td>
                <td><code itemprop="name">Richard Khoury</code></td>
              </tr>
              <tr>
                <td>sameAs</td>
                <td><code itemprop="sameAs">https://scholar.google.com/citations?user=9MrPtC0AAAAJ&hl=en&oi=ao</code></td>
              </tr>
            </table>
          </div>
        </td>
      </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">GRAIL</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://grail.ift.ulaval.ca/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">CC-BY-NC-SA 4.0</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
    <tr>
    <td>citation</td>
    <td><code itemprop="citation">...</code></td>
  </tr>
