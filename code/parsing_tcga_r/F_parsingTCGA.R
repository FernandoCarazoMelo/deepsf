# Input
#   TCGA_tpm_rd[1:5,1:2]
# # A tibble: 5 x 2
# X1                                                                                  `bb9fff02-9a50-4b97-90d8-daecf~
#   <chr>                                                                                                         <dbl>
#   1 ENST00000000233.9|ENSG00000004059.10|OTTHUMG00000023246.6|OTTHUMT00000059567.2|ARF~                          146   
# 2 ENST00000000412.7|ENSG00000003056.7|OTTHUMG00000168276.2|OTTHUMT00000399130.1|M6PR~                           35.8 
# 3 ENST00000000442.10|ENSG00000173153.13|OTTHUMG00000150641.6|OTTHUMT00000319303.1|ES~                           24.1 
# 4 ENST00000001008.5|ENSG00000004478.7|OTTHUMG00000090429.3|OTTHUMT00000206861.2|FKBP~                           50.7 
# 5 ENST00000001146.6|ENSG00000003137.8|OTTHUMG00000129756.5|OTTHUMT00000251969.1|CYP2~                            3.50
# 
# TCGA_ID_MAP[1:5,]
# CGHubAnalysisID               AliquotBarcode                           Aliquot_id Disease  Sample_Name sampleType
# 1 bb9fff02-9a50-4b97-90d8-daecfa9104a0 TCGA-CQ-6229-01A-11R-1915-07 36c026d9-a1a2-4a84-8ee9-eeb708293793    HNSC TCGA-CQ-6229        01A
# 2 0eeec485-9070-4099-b6a0-be5158323751 TCGA-CQ-6227-01A-11R-1915-07 85b4c113-4bf1-4340-a363-37822173e07c    HNSC TCGA-CQ-6227        01A
# 3 f254869c-d6e3-45b7-8144-20be5030a3f1 TCGA-CQ-6225-01A-11R-1915-07 36c2684d-5952-4705-abda-655df65b246c    HNSC TCGA-CQ-6225        01A
# 4 f3dca814-de43-4984-ba7a-0641ac206f93 TCGA-CQ-6218-01A-11R-1915-07 ffbf22e0-3863-4e8d-95a4-99e1c05690b7    HNSC TCGA-CQ-6218        01A
# 5 22acd051-cc4f-4aa3-a91b-83787ebbdd78 TCGA-CQ-6219-01A-11R-1915-07 b2b301e9-2e77-4f62-838d-dba96fab164d    HNSC TCGA-CQ-6219        01A

parsingTCGA <- function(TCGA_tpm, TCGA_ID_MAP){
  
  TCGA_tpm <- as.data.frame(TCGA_tpm)
  rownames(TCGA_tpm) <- TCGA_tpm$X1
  X1 <- TCGA_tpm$X1
  TCGA_tpm <- TCGA_tpm %>% dplyr::select(-X1)
  
  # No duplicated cols
  # which(duplicated(colnames(TCGA_tpm)))
  # TCGA_tpm <- TCGA_tpm[, -which(duplicated(colnames(TCGA_tpm)))]
  
  # Change colnames: AliquotBarcode
  iT <- match(colnames(TCGA_tpm), TCGA_ID_MAP$CGHubAnalysisID)
  if(length(which(is.na(iT))) > 0) stop("NAs found")
  colnames(TCGA_tpm) <- TCGA_ID_MAP$AliquotBarcode[iT]
  TCGA_ID_MAP <- TCGA_ID_MAP[iT, ]

  # Duplicated
  iDup <- which(duplicated(colnames(TCGA_tpm)))
  if (length(iDup) != 0) {
    TCGA_tpm <- TCGA_tpm[,-iDup]
    TCGA_ID_MAP <- TCGA_ID_MAP[-iDup,]
  }
  
  # get sample names and types
  nms2 <- mapply(strsplit(x = colnames(TCGA_tpm), split = "-"), FUN = function(x){paste(x[1:3], collapse = "-")})
  sampleType <- mapply(strsplit(x = colnames(TCGA_tpm), split = "-"), FUN = function(x){paste(x[4], collapse = "-")})
  table(sampleType)
  TCGA_ID_MAP <- cbind(TCGA_ID_MAP, Sample_Name = nms2, sampleType)
  
  # Transcripts info --------------------------------------------------------
  
  auxL <- strsplit(X1,"[|]")
  tID <- unlist(strsplit(x = mapply(auxL, FUN = function(x){return(x[1])}), split = "[.]"))[c(T,F)]# transcript_ID
  gID <- unlist(strsplit(x = mapply(auxL, FUN = function(x){return(x[2])}), split = "[.]"))[c(T,F)]# gene_ID
  tN <- mapply(auxL, FUN = function(x){return(x[5])})# transcrip_name 
  gN <- mapply(auxL, FUN = function(x){return(x[6])})#gene_name
  bT <- mapply(auxL, FUN = function(x){return(x[8])})# biotype
  getBM <- data.frame(Transcript_ID = tID,
                      Gene_ID = gID,
                      Transcrip_name = tN,
                      Gene_name = gN,
                      Biotype =bT,
                      stringsAsFactors = F)
  
  rownames(TCGA_tpm) <- (getBM$Transcript_ID)
  
  # Summarize by genes
  TCGA_tpm_gn <- cbind(gene_ID = getBM[,c(2)], TCGA_tpm)
  TCGA_tpm_gn <- TCGA_tpm_gn %>% group_by(gene_ID) %>% summarise_each(funs(sum))
  rownames(TCGA_tpm_gn) <- unlist(TCGA_tpm_gn[,1])
  TCGA_tpm_gn <- TCGA_tpm_gn %>% dplyr::select(-gene_ID)
  TCGA_tpm_gn <- as.data.frame(TCGA_tpm_gn)
  iG <- match(rownames(TCGA_tpm_gn), getBM$Gene_ID)
  getBM_gn <- getBM[iG,]
  rownames(TCGA_tpm_gn) <- make.unique(getBM_gn$Gene_name)
  
  
  # Splitting samples into tumor and normal ---------------------------------
  
  # Solid Tissue Normal
  iNormal <- which(sampleType == "11A"  | sampleType == "11B")
  
  # # Recurrent solid tumor
  iRecurrent <- which(sampleType == "02A")
  
  # Primary Solid Tumor
  iTumor <- which(sampleType == "01A" | sampleType == "01B" | sampleType == "01C")
  
  # sampleNames <- colnames(TCGA_tpm)
  if(length(which(duplicated(nms2[iTumor]))) != 0) iTumor <- iTumor[-which(duplicated(nms2[iTumor]))]
  
  
  ###################
  # colnames(TCGA_tpm) <- nms2
  
  # rm(iT, nms2, TCGA_ID_MAP)
  
  # TCGA_tpm_NORMAL <- TCGA_tpm[, iNormal]
  # TCGA_tpm_TUMOR <- TCGA_tpm[, iTumor]
  # save(TCGA_tpm_NORMAL, TCGA_tpm_TUMOR, getBM)
  
  cond <- rep(NA, ncol(TCGA_tpm))
  cond[iNormal] <- "Normal"
  cond[iTumor] <- "Tumor"
  iOther <- which(is.na(cond))
  cond[iOther] <- "Other"
  cond <- as.factor(cond)
  cond <-relevel(cond, "Normal")
  # table(cond)
  
  colnames(TCGA_tpm) <- colnames(TCGA_tpm_gn) <- paste(colnames(TCGA_tpm), cond, sep = "_")
  
  return(list(TCGA_tpm = TCGA_tpm, TCGA_tpm_gn = TCGA_tpm_gn, cond = cond, getBM = getBM))
}