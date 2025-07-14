# Translation Efficiency

## Overview

- [github_issue](https://github.com/ManchesterBioinference/mRNA_LLM/issues/1)
- Translational efficiency (TE) refers to an increase in rate of protein synthesis per unit of mRNA, presumably through an increased number of ribosomes bound to a single mRNA. [source](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/translational-efficiency#:~:text=capacity.-,Translational%20efficiency%20refers%20to%20an%20increase%20in%20rate%20of%20protein%20synthesis%20per%20unit%20of%20mRNA%2C%20presumably%20through%20an%20increased%20number%20of%20ribosomes%20bound%20to%20a%20single%20mRNA.,-Translational)
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
  <summary><B>2025.06.03 - grid search complete with introns removed (worse than when they are retained)</B></summary>

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

  - [X] I should probably test it with a few different sequences to see if that changes the results or use random sequences keeping the GC content the same.

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

- Look at the variability in each of the components to determine which one has the most impact on the prediction.
  - sequence: 0.1201
  - extraFeatures: 0.1048
  - 5'UTR: 0.0752
  - 3'UTR: 0.0840
- I thought 5'UTR was supposed to have a bigger impact than the 3'UTR, but it appears that the 3'UTR has a bigger impact. I will have to look into this more.

- What to do next:
  - [ ] review the motif hits
  - [ ] is there a better motif database to use? should I be looking at microRNA binding sites?
  - [ ] what is an example of something we already know in the TE space that we can use to validate the model?
    - [ ] Can I find a transcript that has already been well studied in the context of TE and see if the model extracts the same features as the literature?
  - [ ] Is there a certain combination of RBPs that are important for TE according to the data?

- Motif Hits
  - No Negative Hits
  - Positive Hits
    - RNP4F: [lit review by LLM](https://chatgpt.com/share/686b87f4-73bc-8003-b639-a3d19deba145)
    - MOD: [lit review by LLM](https://chatgpt.com/share/686b979b-4e88-8003-82de-978ddb13503d)

- RBP combinations
  - I've found RBPs that are significantly co-occurring with RNP4F, but it could simply be that the have such similar binding motifs that they are finding the same sequence and competing rather than working together. How can i correct for this?
    - I used TomTom to compare the motifs and rank their similarity.
      > singularity exec docker://memesuite/memesuite:5.5.7 bash --login
      > tomtom -thresh 1 -m RNCMPT00060 /opt/meme/share/meme-5.5.7/db/motif_databases/RNA/Ray2013_rbp_Drosophila_melanogaster.meme /opt/meme/share/meme-5.5.7/db/motif_databases/RNA/Ray2013_rbp_Drosophila_melanogaster.meme

</details>

<details>
  <summary><B>2025.07.08 - investigate oskar (transcript with known loops influencing localization and translation rate)</B></summary>

- In my attempt to find a transcript that has already been well studied in the context of TE, I found the oskar transcript. It is known to have a loop structure that influences its localization and translation rate. I will use this transcript to validate the model and see if it extracts the same features as the literature.
- The 5'UTR is only "GGAUCACUUUCCUCCAAGCG", so I plugged just the 3'UTR into the ViennaRNA to predict and visualize the secondary structure. 

![ViennaRNA structure](../../figures/rna.svg)

- According to this [manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046350/), the green box is the SL2a loop most likely responsible for Stuafen binding (SRS Staufen recognized structures, double stranded RNA) that controls localization.
- The blue box is the small region that has a negative SHAP score. The rest of the transcript has a low positive SHAP score, which makes a bit more sense when I see that the whole thing is bound in loops. I wonder if there is anything important about the blue loop that relates to TE.

- I'm not sure oskar is going to be the best example. Doing an [LLM search](https://chatgpt.com/share/686d1a14-888c-8003-83ff-5c4d1537f1a7) shows that these transcripts have been studied in detail in relation to TE.

  | In training set | Name | FlyBase ID         | Paper |
  |-----------------|------------------------|----------------|-------|
  | [ ]             | msl-2 (male-specific lethal-2) | FBgn0005616   | [link]() |
  | [X]             | osk (oskar)            | FBgn0003015    | [link]() |
  | [X]             | cad (caudal)           | FBgn0000251    | [link](https://www.nature.com/articles/379694a0) |
  | [X]             | nos (nanos)            | FBgn0002962    | [link](https://genesdev.cshlp.org/content/10/20/2600.full.pdf) |
  | [X]             | hb (hunchback)         | FBgn0001180    | [link](https://pdf.sciencedirectassets.com/272196/1-s2.0-S0092867400X03728/1-s2.0-0092867491903689/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCxqnfDeqT8AIqn4ZPAg2KLnTEKqMb2TlMoDQw8zBzNPAIhAK4kL7%2BPxwtcGF%2Fr3rdtFAcnicL1DiU0Bwe9EXbTQ7E8KrsFCKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgxMBulAE%2BVqtvfQt18qjwW4NBqk%2BFnULyd11h3S76CAv19iemklJkEqnJW1dVccrNW2P2O3BGWg2SpoMpy%2FjXYF5HTbPRpTK24sWCdmgl7DeE51mJTAUpCHUtUgd1fQWJYoek3OxdKYzzN4W61v%2FbxjfXnfAI8rGtyDbHhME%2BSqbSXqs%2BHUBVfirzqJynDhkO7lQ%2B8Pze2nwR2jeHNErEAwyZNEZWHKS2V4SkW%2Fr8jnq1MWC3%2FknG1km0A3D1JqptHCKBoP6biNtreLPUkIHMNnMiBf%2BxYPD3qRV90Ed71KFfo7%2BRRLUiP%2BNy2mePNHkgsd%2BTVJeHSL13eVohEbWxnhwDjzI7lmJa0wASG%2Brliw1Vu83OHRqme88GPxXQUShvrZqKnrHJU2qvH1rTvwn1lO8Or3%2BGqx3l9%2Bx0I%2BQNXJSkXu%2F5mXpepQ2MWWJ95wZGLiKF%2FCOEsKDLBkDpdQbbvTeGMvdPCSuPDxjAsRapX%2FpdUH%2B%2FFYyr2nDCcFOCRIad1QmM1npPOKyF7Z0W4m%2FsqETP2VeYpH496fxZ%2BG0jsYRgTbBAK2hvscqCU7XrsjaIgfmbCTWpgsMwjTvtzZchmhu8ZcnvDsEVhA9oJ837bSw0nBUnT0S5%2BwzUJu%2FxAD5FexNGHXgZKKHY3jDGJ%2BCOhPnWPFNbXwmDmZEyxxb7O1H%2ByV%2B9WMUAszn8cgTi6poOFcdZcSUx%2BXx8vxKNjy8sIZ74VBLXjwWRxqQF4f8w7kpRZQPng2oZ9RrytDkYh51lT12xcAMnWb65LRNGhnqDGE1wlV4MwKf07TTcBgCuWsk%2BeXGxeO%2BZ4%2BppWxKW3tMAVklj0VXA2LHFNc3rVa8b6t2ZMofJQ8Zmwu5JxhycMLyleeseAklRepTzAgW6BCMOeluMMGOrAB3F1iG3sAL7wmz0Lf1Ext094WvUKKeh%2BOcowz1JySAXieM4dVYuPgQypFw5gcAtmbRlsNUA8WiNBdc3BLPbWXF%2FFs8ueAxLnUsbTbvmz2Eunq0yH00bjmLvQ3GbRl4cPGLFFGeX6S0ADwZ35n9bhv%2B0o5AvxxDGO3f9ymU7spK3odXPZXeTb84CbB5PSsdxT0h8p3PUbQHgDP7ZFjELWLwhPeV%2F6uZu%2BxT%2BM8%2F%2Fa8ktg%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250709T072528Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUGHUFUXG%2F20250709%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7373d1d8ae4bb55be85440a1a9e20c0597cbe7371345ba8a48ba201496227010&hash=041b691301b81b7e50e3d084fd113a0eedaf1663aa8187c19564eb0b33fb4784&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0092867491903689&tid=spdf-29b18a31-9d64-4211-b9e4-0d4e73c9d1db&sid=6831636340e8464750481a96dc985f0c52c0gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080058510458515858&rr=95c6166aea6c0765&cc=gb)|

</details>

<details>
  <summary><B>2025.07.09 - Model validation on known (hb, nos, cad) translationally regulated transcripts</B></summary>

### hb

- hb is known to be translationally repressed by two Nanos Response Elements (NREs) in the 3'UTR. Consenus (GUUGUnnnnnAUUGUA)
  - NRE 1: "GUUGUCCAGAAUUGUA"
  - NRE 2: "GUUGUCGAAAAUUGUA"

![Importance visualization with highlighted NREs](../../figures/hb_NREs.svg)

- It appears that the model is recognizing at least one of the NREs as leading to a **negative** impact on the TE value. This is a good sign that the model is learning the biology of the system.

### nos

- nos is known to be translationally repressed by the Smaug Response Elements (SRE) in the 21-80 nucleotide region of the 3'UTR.
  - SRE: "GAGCAGAGGCUCUGGCAGCUUUUGCAGCGUUUAUAUAACAUGAAAUAUAUAUACGCAUUCC"
  
![Importance visualization with highlighted SRE](../../figures/nos_SRE.svg)

- The model is showing this has a **positive** impact on the TE value, which is not what we expect. The model was pretty far off on its prediction.

### cad

- cad is translationally repressed by the bcd recognition element (BRE) in the 3'UTR (210-553 nucleotide region).
  - BRE: "ACCGAACCGAAAAGUUAAUAGGCAGCCGGACGAAUGGAGGACUUGGCGGCCGUUGCACCUGGAAUAUUGCACGUUGUUAAUUUUUGUGAUUGUAUAUUCCUGGUUUCGACACGCGCCAGAGUCCUCACAGCUAAACAAGUCUUAUAUUAUUCUUGUAUUAUGUUUGUUUUUUGUUCAACGUGUGUAGUAGCUUAAAGUAAAAUGAAUAGCUCGUAAGCAGUAGUAAGUAAAGUUGCCCGAGAAAAACAAGAACAAUUCAAACCAGCCGUCCCAGCCGAUUAACGUUUAAAAGUACUCGCUGCAGUUAAACAUAAUUUUAGUACAAGCAACUCAUUUUAGAGCG"

![Importance visualization with highlighted BRE](../../figures/cad_BRE.png)

- This whole region plus a bit on each end is the BRE. This region has lower scores than most of the transcript, but still has several regions of **positive** impact on the TE value.

</details>

<details>
  <summary><B>2025.07.10 - </B></summary>

- What to do next:
  - [ ] review the motif hits
  - [ ] is there a better motif database to use? should I be looking at microRNA binding sites?
  - [ ] what is an example of something we already know in the TE space that we can use to validate the model?
    - [ ] Can I find a transcript that has already been well studied in the context of TE and see if the model extracts the same features as the literature?
  - [ ] Is there a certain combination of RBPs that are important for TE according to the data?

</details>

<details>
  <summary><B>2025.07.11 - </B></summary>

- I have been attempting to run a co-occurrence analysis on the predicted motifs, but was running into issues. I ended up running the data through a FIMO analysis to get the exact locations of the motifs that way i could exclude instances where the secondary motif was competing for the same location. [exploreRNP4F.ipynb](../../notebooks/explorativeJupyterNotebooks/exploreRNP4F.ipynb)

- I also compared the highLow TE AME run results to the LLM_SHAP TE AME results [compare_highLow_to_LLMshap.ipynb](../../notebooks/explorativeJupyterNotebooks/compare_highLow_to_LLMshap.ipynb). The results are quite different, which could be interesting if we could find a way to prove that one method is better than the other.

- gold standards to compare to:
  - [ ] kmer SVM: did this for the RBP binding already, but would need to switch to svm regression. The package should be able to handle this.
  - [ ] run a linear regression model that includes the extra features and the **counts for predicted motifs** (67 motifs from the database).

</details>

<details>
  <summary><B>2025.07.14 - linear regression: TE prediction using RBP counts and extraFeatures</B></summary>

- So I implemented the linear regression model so we can determine if the LLM or the high/low analysis provides better RBP results. To get the RBP counts, I used MAST. MAST allows you to remove highly similar motifs so you don't skew your results. I realized that this same approach is not included in AME and is not even a feature in AME. I now run MAST before AME, so i can filter the motif database which makes the comparison between the two methods more fair (same motifs used across the board). 
- interestingly, the low vs high has a much better correlation with the LLM method than the high vs low. this is strange because at first thought the LLM positive and high vs low should both be identifying the motifs associated with a higher TE score. Yet, the correlations are reversed. 
  - high vs low & LLM: spearman (-0.13), pval (0.50)
![high & LLM scatterPlot](../../figures/highVSllm.png)
  - low vs high & LLM: spearman (0.27), pval (0.19)
![low & LLM scatterPlot](../../figures/lowVSllm.png)
  - Is this because the low and high TE mRNA are controlled differently and even though we are selecting for LLM regions that increase the TE, they are regions that increase the TE for the low TE mRNA specifically rather than both (globally)? Could it be that the low TE mRNA provided more positive LLM regions than the high TE mRNA? 
  - We are including a bunch of extra features like GC content, MFE (secondary structure), utr length, and codon frequencies. Could it be that These features are the most important for predicting high and low TE, but then the RBP motifs are fine-tuning it from there? that potentially, these motifs are associated with Low TE, but they are making these low TE have higher TE than they would otherwise?
- Either way, the LLM is very correlated with the linear regression model, especially when compared to the high vs low & regression model.
  - Could this be a result of including the extra features in the LLM and regression model, where the high vs low only includes the sequence?
![regression & LLM scatterPlot](../../figures/linearVSllm.png)

</details>

<details>
  <summary><B>2025.07.</B></summary>

-

</details>
