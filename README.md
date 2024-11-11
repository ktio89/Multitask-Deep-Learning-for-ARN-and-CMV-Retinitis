# Multitask Deep Learning for Joint Detection of Necrotizing Viral and Noninfectious Retinitis From Common Blood and Serology Test Data

---

The Official Repository of the paper **"Multitask Deep Learning for Joint Detection of Necrotizing Viral and Noninfectious Retinitis From Common Blood and Serology Test Data"**.\
This paper is accepted to **IOVS (Investigative Ophthalmology & Visual Science) 2024** (Q1; IF = 5.0).\
Authors: Kai Tzu-iunn Ong; Taeyoon Kwon; Harok Jang; Min Kim; Christopher Seungkyu Lee; Suk Ho Byeon; Sung Soo Kim; Jinyoung Yeo; Eun Young Choi\
Paper Link: https://iovs.arvojournals.org/Article.aspx?articleid=2793342

## Data

Due to privacy and ethical issues, we cannot share participant data here.\
However, since adopted clinical features are listed in our paper, you are able to accordingly prepare your collected blood data for implementation.\
Questions regarding accessing our data should be directed to the co-corresponding author: Eun Young Choi (eychoi@yuhs.ac).

## Requirements

## Implementations

**utils/dataset.py**: Once you finished preparing your own data or obtained data from us, use this to prepare the data for training/testing.
**utils/model.py**: This file include the following models: a Base DL model (i.e., Single Task MLP), Fully-shared Multi-task Learning model (FSMTL), Shard-private MTL model (SPMTL), and Adversarial MTL model (ADMTL).
**scripts/train.sh**: Training all 5 models (there will be 2 Base DL models. One for ARN detection, and one for CMV retinitis detection).
**scripts/test.sh**: Run inference on all 5 models.

## Cite this work

If you find this paper helpful, please use the following BibTeX to cite our paper, thank you!

```
@article{ong2024multitask,
  title={Multitask deep learning for Joint detection of necrotizing viral and noninfectious retinitis from common blood and serology test data},
  author={Ong, Kai Tzu-iunn and Kwon, Taeyoon and Jang, Harok and Kim, Min and Lee, Christopher Seungkyu and Byeon, Suk Ho and Kim, Sung Soo and Yeo, Jinyoung and Choi, Eun Young},
  journal={Investigative Ophthalmology \& Visual Science},
  volume={65},
  number={2},
  pages={5--5},
  year={2024},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```
