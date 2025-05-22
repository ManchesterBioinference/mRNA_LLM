# Translation Efficiency

## Overview

- [github_issue](https://github.com/ManchesterBioinference/mRNA_LLM/issues/1)
- The translation efficiency (TE) of a gene is defined as the ratio of the number of ribosomes bound to the mRNA to the number of ribosomes that could potentially bind to the mRNA.
- This metric is important for understanding how efficiently a gene is translated into protein.
- In this project I will use the TE data provided by Declan Creamer.
- It has three replicates for each gene and has three different time points.

## Daily Notes

<details>
  <summary><B>2025.05.20 - map genes to transcripts</B></summary>

The data is by gene ID (flybase) but my previous analysis were done by transcript ID. I need to figure out the best way to get the best transcript ID for each gene.

- [ ] Ask Declan if a specific transcript ID was used to for each of the genes.

For now I will use the transcript level expression data that I have from Mike to identify which transcript is the most highly expressed for each gene.
[script](../scripts/mergeUTRsAndDecayRates.py)

- [ ] figure out what thresholds to use for the TE data [script](../scripts/mergeUTRsAndDecayRates.py) line 55

- I should probably explore the data a bit more before I start filtering it. I will plot the distribution of the TE values for each gene and see if there are any outliers.

</details>


<details>
  <summary><B>2025.05.21 - ViennaRNA, TE filters, transcript level expression</B></summary>

- I think I should use the ViennaRNA package to calculate the secondary structure and MFE of the mRNA. This will help me understand how the mRNA is folded and how this affects the translation efficiency. I can add these features to the classification head.
- added a comment about TE value filtering to the github issue.
- added code to the [script](../scripts/mergeUTRsAndDecayRates.py) to find the most highly expressed transcript for each gene. I used the transcript level expression data from Mike to do this. 
  - some transcripts have no expression data. ***I will just pick the first one in these cases.*** [script](../scripts/mergeUTRsAndDecayRates.py) line 55

</details>

<details>
  <summary><B>2025.05.22 - ViennaRNA script</B></summary>

- start working on the ViennaRNA script.
- I realize I need to also create a script that isolates the full transcript. right now I'm just extracting the UTRs and the CDS. I need to get the full transcript sequence for the ViennaRNA package.

</details>
