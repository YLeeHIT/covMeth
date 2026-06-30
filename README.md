# covMeth
 Signal filtering–based smoothing and imputation of sparse CpG methylation profiles

 **Classification-based inference of low-coverage CpG methylation for haplotype-resolved DMR detection**

 covMeth is a computational method for recovering methylation levels at low-coverage CpG sites in haplotype-resolved DNA methylation data. It classifies low-coverage CpG sites and regions according to their spatial distribution and local methylation context, and then applies a structure-specific inference strategy to improve the accuracy and stability of haplotype-resolved differentially methylated region (**hDMR**) detection.

 ---

## Overview

 Haplotype-resolved DNA methylation analysis divides sequencing reads between two haplotypes, which further reduces the effective coverage available for estimating CpG methylation levels. Low-coverage CpG sites may therefore produce unstable methylation estimates, interrupt genuine allele-specific methylation patterns, and lead to fragmented, missed, or falsely identified hDMRs.

 covMeth addresses this problem using a classification-based framework. Instead of applying a single smoothing or imputation model to all low-coverage CpG sites, covMeth divides them into three representative structural patterns and performs targeted methylation inference for each pattern:

 1. **Isolated low-coverage CpG sites**
 2. **Local low-coverage CpG blocks bounded by high-coverage anchors**
 3. **Continuous low-coverage CpG regions without reliable nearby anchors**

 The corrected haplotype-specific methylation matrix can subsequently be used by downstream hDMR detection tools such as cyberDMR.

 ---

## Key features

 - Classifies low-coverage CpG sites according to their genomic spatial structure.
 - Uses different inference strategies for isolated sites, local blocks, and continuous low-coverage regions.
 - Integrates CpG methylation level, sequencing depth, genomic distance, and local consistency.
 - Preserves available observations rather than replacing all low-coverage measurements directly.
 - Reduces artificial interruption and fragmentation of genuine hDMRs.
 - Limits excessive smoothing across real methylation boundaries.
 - Supports haplotype-resolved long-read methylation data.
 - Has linear time complexity with respect to the number of low-coverage CpG sites.
 - Maintains high hDMR detection accuracy under low sequencing coverage.

 ---

