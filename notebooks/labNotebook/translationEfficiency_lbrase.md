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

<details>
  <summary><B>2025.05.27 - filter TE values</B></summary>

- After talking to Magnus we decided to find all the genes/transcripts with 0 mRNA values and set them all to the same max value. After identifying these transcripts, I realized that not all 0 mRNA counts led to a high TE value, which makes sense because the ribo counts can also be low. I've decided to go with a fairly conservative approach and only keep transcripts that have at least 1 count in each replicate for both the mRNA and ribo data.
- I have also kept the double log transformation of the TE values, in the hopes that this will help focus the model on the bulk of the data and not the outliers.

</details>

<details>
  <summary><B>2025.05.28 - allow runViennaRNA.py to run in parallel</B></summary>

- I now properly pass only the transcripts that have been filtered to the runViennaRNA.py script, so save some time.
- I also added a feature to allow the script to run in parallel, which should speed up the process of calculating the secondary structure and MFE of the mRNA.
- once I have the ViennaRNA data, it should rerun the fineTuneModel.py script. Hopefully, it improves the performance of the model.

</details>

<details>
  <summary><B>2025.05.29 - best method for grid search, suggestions from Hilary and William </B></summary>

- What to accomplish today:

  - Check on the run that included ViennaRNA results.
  - run grid search on the vienna+ model and the original model to see which one performs better.
    - need to figure out how to make the dvc experiments work. right now the experiment disappears after the run is done. It is an issue with running it over slurm.
      - A known work around is to push the experiment to the remote repository after the run is done. I don't want to do this because it will create a lot of noise in the repository. I will try to figure out a way to keep the experiment locally.
      - i could also just write the best global spearman correlation to a file like i was doing before. Just make sure to include the hyperparameters used for the run, so i can reset it to those and run again to load the saved outputs.
- I just had a meeting with Hilary and William. Hilary suggests that the second time point is the most interesting one to look at (I'm currently looking at the first time point). William also suggested that i should increase the mRNA threshold to 10 counts instead of 1 because dividing my a small number can lead to very large TE values, which can skew the results. I will think about this....

</details>

<details>
  <summary><B>2025.05.30 - run on time point 2, start grid search</B></summary>

- I started a grid search yesterday, but I think I didn't swap over to time point 2. I will check on that today.
  - I did forget. thankfully, I didn't start the grid search yet. I still need to run the ViennaRNA script on the time point 2 data, so I will do that first. I run it separately with 8 cores to speed it up and then run the rest of the grid search using 2 L40S gpus.
- Higher level plan for the project:
  - use LLMs to predict TE values
    - look at the features that are most important for the model
    - compare the amount of importance between the 5' and 3' UTRs
  - use LLMs to predict zygotic decay rates
    - look at the features that are most important for the model
    - compare the amount of importance between the 5' and 3' UTRs
  -

</details>

<details>
  <summary><B>2025.06.02 - remove introns for viennaRNA run, deseq2, dvclive image cache</B></summary>

- run ViennaRNA on the 5'UTR + CDS  + 3'UTR sequences not the intron inclusive sequences.
- do some research into deseq2 and if it has been used with TE before.
- I just learned that in order for dvclive to cache the images, i need to run live.end(). so all the plots have been saved to git up to this point. it has increased the size of the repo by over 100MB. Hopefully, this fix will now keep the size down.

</details>

<details>
  <summary><B>2025.06.03 - grid search complete with introns removed</B></summary>

- interestingly the results are a bit worse than the previous run with the introns included. I will have to look into this more.
  - best run: spearman (0.49949925064783884) - 50 epochs, 5e-5 learning rate, 0.1 warmup percent
  - best run with introns: spearman (0.5094562418533073) - 40 epochs, 0.0001 learning rate, 0.15 warm up percent

</details>

<details>
  <summary><B>2025.06.04 - DESeq2 dispersion, single log transform without adding 1</B></summary>

- over the last couple of days I got DESeq2 working. I used it to calculate the corrected dispersion for the riboSeq and rnaSeq individually. Removing all genes with a dispersion value > 1 removes a few hundred more genes than the 1 count threshold I was using before. I'm not convinced it is different enough to warrant the extra complexity. I will stick with the 1 count threshold for now.
  - LLM discussion about the DESeq2 dispersion: [link](https://grok.com/share/bGVnYWN5_5adfabad-17c7-482b-91f9-8ff01fef38be)
- During this process I realized that the TE data is all > 0. this means i don't need to add 1 when doing the log transformation and makes the data extremely normalized from a single log transformation rather than a double log transformation. I am now doing a test run using the best hyperparameters from the grid search.
  - This approach led to a distribution that was too tall and skinny (too much kertosis). I'll stay with the double log transformation for now.

- [X] I need to adjust the findMotifs and runAME scripts to only use the training data that it successfully predicted on. According to Mikhail, this will prevent pulling features into the motif identification that led to incorrect predictions.
  - how exactly do i classify a prediction as correct or incorrect in a regression task? I was going to use a threshold on the difference, but I'm not so sure about that because error is not uniform across the range of TE values. the low values are over predicted and the high values are under predicted.
  > The solution I went with was to train a LOWESS regression model on the residuals. I then removed the samples that were more than 2 standard deviations away from the LOWESS regression line. This should remove the samples that are not predicted well by the model.

</details>

<details>
  <summary><B>2025.06.05 - run rest of pipeline (determine if extra features or sequences and 5'UTR or 3'UTR has a bigger influence on prediction)</B></summary>

- I ran the rest of the pipeline which includes randomizing the sequences, extra features, 5'UTR and 3'UTR. It appears that the extra features have the biggest impact, then the 5'UTR, then the full sequence, and finally the 3'UTR. The full sequences makes sense because it includes the 5'UTR so should be between them right?

  - hmmm. maybe not. you could have predicted that the full seq would be better than the 5'UTR alone because of the extra information, but it is not. Why would swapping the 5'UTR by itself have a bigger impact that swapping the full sequence? I bet some of it is because I only tested it with one sequence, as in I used the same 5'UTR/3'UTRs for all the transcripts.

  - [ ] I should probably test it with a few different sequences to see if that changes the results or use random sequences keeping the GC content the same.

</details>

<details>
  <summary><B>2025.06.06 - 5 vs 3 UTR effect, importance visualization threshold</B></summary>

- plan for the day:
  - [ ] try to interpret the motif results (is that AME database the best to use?)
  - [ ] run a statistical test to see if the extra features are significantly better than the 5'UTR and 3'UTR
    - I additionally, updated the script to run every combination of features and sequences to avoid any bias in the results. This should make the statistical tests more robust
  - [X] review the importance visualization. All the colors are muted so i want to check if there are some outliers that should be thresholded to improve the visualization of the other values.
    - I set a threshold of mean + 4*std for the importance values. This should help to visualize the important features better.

</details>

<details>
  <summary><B>2025.06.13 - start run for all combinations of sequences, extraFeatures, 3UTR, and 5UTR to determine each components impact on prediction</B></summary>

- when running every combination of features and sequences, I now skip sequences that were too long for the model. I also correctly copy the records so that the original data is not modified during the nested loop.
- I didn't realize how long it would take to run the every combination. We have 4240 transcripts, so that is 4240^2 combinations (18 million) and I'm running that four different times (seqImpact, extraFeatImpact, 5UTRImpact, 3UTRImpact). At 206,080 iterations in 10min that will be 14.5 hours per run or 58.25 hours total. Good thing I can run it over the weekend.
- I have RNP4F binding to the 3'UTR not the 5'UTR

</details>

<details>
  <summary><B>2025.06.30 - fix a bug in extraFeatures impact</B></summary>

- I found a bug in the way the extra features impact was being prepared. It was including the old and new sequence ID which made the transcript ID not fall in the correct position. I fixed this and reran the extra features impact. 

</details>

<details>
  <summary><B>2025.07.07 - </B></summary>

- I now look at the variability in each of the components to determine which one has the most impact on the prediction.
  - sequence: 0.1201
  - extraFeatures: 0.1048
  - 5'UTR: 0.0752
  - 3'UTR: 0.0840
- I thought 5'UTR was supposed to have a bigger impact than the 3'UTR, but it appears that the 3'UTR has a bigger impact. I will have to look into this more.

</details>
