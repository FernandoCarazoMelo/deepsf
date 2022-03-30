
####################################################
# Author: Fernando Carazo | fcarazo@tecnun.es
# Date: 2021-12-23
####################################################

library(readr)
library(dplyr)
library(TCGAbiolinks)
library(DT)
library(readxl)
library(limma)
library(matrixStats)
library(Matrix)
library(biomaRt)
library(pheatmap)
library(survival)
library(psych)
library(ggplot2)

source("./code/external/ggmatplot.R")
source("./code/external/multiplot.R")
source('./code/external/GetPSI_Kallisto.R') 
source("./code/F_parsingTCGA.R")


# Variables ---------------------------------------------------------------

cancerSite <- "HNSC"
pathOutput <- "Z:/BACKUP_GRUPO/FernandoCarazo/ProyectosCompletos/stanford-research/data/output/TCGA"
dirResults <- "Z:/BACKUP_GRUPO/FernandoCarazo/ProyectosCompletos/stanford-research/data/output/results"

dirOutput <- sprintf("%s/%s", pathOutput, cancerSite)


parsing <- FALSE
getPSI <- FALSE
plotEv <- TRUE
modEvF <- FALSE #(run only one time)
# eventsPath <- "./data/input/EventsFound_genecode24/good_classification/EventsFound_gencode_v24_goodclassification.txt" # Version at Stanford until 2021-02-17
eventsPath <- "./data/input/2021-02-15_EP-bootstrap_JAF/EventsFound_gencode24.txt" # new Version of Juan from 2021-02-17



# parsingTCGA -----------------------------------------------------------

if(parsing){
  TCGA_tpm_rd <- read_delim(sprintf("~/Bunker/Samples/TCGA-Kallisto/%s/TCGA_%s_tpm.tsv", cancerSite, cancerSite), "\t", escape_double = FALSE, trim_ws = TRUE)
  TCGA_ID_MAP <- read_excel("~/Bunker/Samples/TCGA-Kallisto/TCGA_ID_MAP.xlsx")
  ParsTCGA <- parsingTCGA(TCGA_tpm = TCGA_tpm_rd, TCGA_ID_MAP = TCGA_ID_MAP)
  
  TCGA_tpm <- ParsTCGA$TCGA_tpm
  TCGA_tpm_gn <- ParsTCGA$TCGA_tpm_gn
  cond <- ParsTCGA$cond
  getBM <- ParsTCGA$getBM
  
  identical(colnames(TCGA_tpm), colnames(TCGA_tpm_gn)) #TRUE
  
  if(!dir.exists(dirOutput)) {
    dir.create(dirOutput)
    save(ParsTCGA, file = sprintf("%s/Parsing_TCGA_%s.RData",dirOutput, cancerSite))
  }
  # rm(ParsTCGA)
}else{
  load(sprintf("%s/Parsing_TCGA_%s.RData",dirOutput, cancerSite))
}

TCGA_tpm <- ParsTCGA$TCGA_tpm 
TCGA_tpm_gn <- ParsTCGA$TCGA_tpm_gn
cond <- ParsTCGA$cond
getBM <- ParsTCGA$getBM
identical(colnames(TCGA_tpm), colnames(TCGA_tpm_gn)) #TRUE
rm(ParsTCGA)