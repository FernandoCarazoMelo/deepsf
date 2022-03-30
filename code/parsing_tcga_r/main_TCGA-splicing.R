


# Modify EventsFound (run only one time)---------------------------------------------

if(modEvF){
  EventsFound <- read_delim(eventsPath,
                            "\t", escape_double = FALSE, trim_ws = TRUE)
  colnames(EventsFound)[1] <- "EventID"
  ENSG <- sapply(strsplit(EventsFound$EventID, "[.]"), function(x) return(x[1]))
  iG <- match(ENSG, getBM$Gene_ID)
  EventsFound$Gene_name <- getBM$Gene_name[iG]
  EventsFound$Biotype <- getBM$Biotype[iG]
  EventsFound$EventType2 <- NA
  EventsFound$EventType2[EventsFound$EventType == "Alternative 3' Splice Site"] <- "A3"
  EventsFound$EventType2[EventsFound$EventType == "Alternative 5' Splice Site"] <- "A5"
  EventsFound$EventType2[EventsFound$EventType == "Alternative First Exon"] <- "AF"
  EventsFound$EventType2[EventsFound$EventType == "Alternative Last Exon"] <- "AL"
  EventsFound$EventType2[EventsFound$EventType == "Cassette Exon"] <- "CE"
  EventsFound$EventType2[EventsFound$EventType == "Complex Event"] <- "C"
  EventsFound$EventType2[EventsFound$EventType == "Mutually Exclusive Exons"] <- "MX"
  EventsFound$EventType2[EventsFound$EventType == "Retained Intron"] <- "RI"
  EventsFound$EventID2 <- paste(EventsFound$Gene_name, EventsFound$EventNumber, EventsFound$EventType2, sep = "_")
  EventsFound$EventID <- paste(EventsFound$EventID, EventsFound$EventNumber, sep = "_")
  # save(EventsFound, file = "./data/input/EventsFound_genecode24/good_classification/EventsFound_gencode_v24_juan2021-02_mod.RData")
}


# Get PSI -----------------------------------------------------------------

if(getPSI){
  load("./data/input/EventsFound_genecode24/good_classification/PathEvxTranx.RData")
  
  # Rename transcripts
  PathEvxTranx$transcritnames <- sapply(strsplit(PathEvxTranx$transcritnames,"\\."),function(X) return(X[1]))
  
  TCGA_tpm <- as.matrix(TCGA_tpm)
  PSI_List <- GetPSI_Kallisto(PathEvxTranx, TCGA_tpm, Qn = 0.25)
  
  # Change names: SAMD11_2_C
  changeN <- TRUE
  if(changeN){
    load(file = "./data/input/EventsFound_genecode24/good_classification/EventsFound_gencode_v24_goodclassification_mod.RData")
    EventsFound <- EventsFound[, -c(6:11)]
    iPs <- match(rownames(PSI_List$PSI), EventsFound$EventID)
    rownames(PSI_List$PSI) <- EventsFound$EventID2[iPs]
    iExp <- match(names(PSI_List$ExpEvs), EventsFound$EventID)
    names(PSI_List$ExpEvs) <- EventsFound$EventID2[iExp]
  }
  
  PSI<-PSI_List$PSI
  ExpEvs<-PSI_List$ExpEvs
  
  # save(PSI_List, file = sprintf("%s/PSI_List_%s.RData",dirOutput, cancerSite))
  save(PSI_List, file = sprintf("%s/PSI_List_%s_good_classification.RData",dirOutput, cancerSite))
  rm(PSI_List)
}else{
  load("./data/input/EventsFound_genecode24/good_classification/PathEvxTranx.RData")
  load(sprintf("%s/PSI_List_%s_good_classification.RData",dirOutput, cancerSite))
  PSI<-PSI_List$PSI
  ExpEvs<-PSI_List$ExpEvs
  rm(PSI_List)
}


# Analysis od PSI ----------------------------------------------------------------

# load(sprintf("%s/PSI_List_%s.RData",dirOutput, cancerSite))

# PSI2 <- PSI
# PSI2[is.nan(PSI2)] <- 0
# 
# SVD <- svd(PSI2)
# SVD <- svd(TCGA_tpm)
# SVD <- svd(TCGA_tpm_gn)

# df <- data.frame(v2 = SVD$v[,2], v3 = SVD$v[,3], cond); ggplot(df, aes(v2, v3)) + geom_point(aes(color = cond), size = 2) + ggtitle("SVD 2")
# df <- data.frame(v3 = SVD$v[,3], v4 = SVD$v[,4], cond); ggplot(df, aes(v3, v4)) + geom_point(aes(color = cond), size = 2) + ggtitle("SVD 3")
# df <- data.frame(v4 = SVD$v[,4], v5 = SVD$v[,5], cond); ggplot(df, aes(v4, v5)) + geom_point(aes(color = cond), size = 2) + ggtitle("SVD 4")
# df <- data.frame(v5 = SVD$v[,5], v6 = SVD$v[,6], cond); ggplot(df, aes(v5, v6)) + geom_point(aes(color = cond), size = 2) + ggtitle("SVD 5")
# df <- data.frame(v6 = SVD$v[,6], v7 = SVD$v[,7], cond); ggplot(df, aes(v6, v7)) + geom_point(aes(color = cond), size = 2) + ggtitle("SVD 6")
# df <- data.frame(v7 = SVD$v[,7], v8 = SVD$v[,8], cond); ggplot(df, aes(v7, v8)) + geom_point(aes(color = cond), size = 2) + ggtitle("SVD 7")

# 6th and 8th components of SVD(PSI) separate patientients into normal and tumor
# x11(); df <- data.frame(v6 = SVD$v[,6], v8 = SVD$v[,8], cond); ggplot(df, aes(v6, v8)) + geom_point(aes(color = cond), size = 2) + ggtitle("SVD 6-8")


# Limma PSI Statistics ----------------------------------------------------------

library(limma)
Dmatrix <- model.matrix(~cond)
colnames(Dmatrix) <- gsub("cond","",colnames(Dmatrix))

fit <- lmFit(PSI, Dmatrix)
Cmatrix <- t(t(c(0,0,1)))

fit2 <- contrasts.fit(fit, Cmatrix)
fit2 <- eBayes(fit2)

TopT <- topTable(fit2,coef=1, num=Inf, sort.by = "P")
head(TopT)

# volcanoplot(fit2)

# Merge with event info
load("./data/input/EventsFound_genecode24/good_classification/EventsFound_gencode_v24_goodclassification_mod.RData")
EventsFound <- as.data.frame(EventsFound)
EventsFound <- EventsFound[!duplicated(EventsFound$EventID2), ]
TopT_a <- merge(TopT, EventsFound, by.x = "ID", by.y = "EventID2", all.x = T)
TopT_a <- TopT_a[order(TopT_a$P.Value), ]
colnames(TopT_a)[1] <- "Event_ID"
rownames(TopT_a) <- NULL
head(TopT_a)[,c(1:11)]

# Results
X11();hist(TopT$P.Value, 100, xlab = "P-value", main = sprintf("Histogram of splicing events' P-values in %s \n (Statistics based on PSI)", cancerSite))

if(!dir.exists(sprintf("%s/%s",dirResults, cancerSite))) {
  dir.create(sprintf("%s/%s",dirResults, cancerSite))
}

if(!dir.exists(sprintf("%s/%s/splicingEvents/",dirResults, cancerSite))) {
  dir.create(sprintf("%s/%s/splicingEvents/",dirResults, cancerSite))
}

write.csv2(TopT_a, file = sprintf("%s/%s/splicingEvents/TopTable_Events.csv",dirResults, cancerSite))


nrow(PathEvxTranx$ExTP1) # 118830
nrow(TopT) # 31626
sum(TopT$P.Value < 0.001, na.rm = T) # 4785
sum(TopT$P.Value < 0.001, na.rm = T) / nrow(PathEvxTranx$ExTP1) # 0.040 = 4%


# Plot top events ---------------------------------------------------------

ordPlot <- order(cond)
# Just Normal and Tumor
ordPlot <- ordPlot[-which(sort(cond) == "Other")]

if(plotEv){
  for(n in 1:50){
    ev <- TopT$ID[n]
    
    
    p1 <- ggmatplot(ExpEvs[[ev]][ordPlot, c(1,2)] / ExpEvs[[ev]][ordPlot, c(3)]) + 
      ggtitle(sprintf("%s. PSI (Ev: %s)", n, ev), subtitle = " Normal | Tumor") +
      xlab(sprintf("%s Samples (n = %s)", cancerSite, length(ordPlot))) +
      ylab("PSI") +
      geom_vline(xintercept = sum(cond=="Normal"), linetype = 2) +
      theme(axis.text.x=element_blank())  + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                  panel.background = element_blank(), axis.line = element_line(colour = "black"))
    
    p2 <- ggmatplot(log2(1+ExpEvs[[ev]][ordPlot, c(1,2)])) +
      ggtitle(sprintf("%s. Expression (Ev: %s)", n, ev), subtitle = " Normal | Tumor") +
      xlab(sprintf("%s Samples (n = %s)", cancerSite, length(ordPlot))) +
      ylab("log2_TPM+1") +
      geom_vline(xintercept = sum(cond=="Normal"), linetype = 2) +
      theme(axis.text.x=element_blank()) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                 panel.background = element_blank(), axis.line = element_line(colour = "black"))
    pdf(file = sprintf("%s/%s/splicingEvents/%s.PSI_Exp_%s.pdf",dirResults, cancerSite, n, ev), width = 10, height = 6)
    multiplot(p1,p2,cols = 1)
    dev.off()
  }
}



# Event type --------------------------------------------------------------

PvThr <- 0.001
TopT_a_s <- TopT_a[which(TopT_a$P.Value < PvThr), ]

sort(table(TopT_a_s$EventType2), decreasing = T)
round(prop.table(sort(table(TopT_a_s$EventType2), decreasing = T))*100,2)
pie(table(TopT_a_s$EventType2))


# IGV track ---------------------------------------------------------------

chr <- unlist(strsplit(TopT_a_s$GPos, ":"))[c(T,F)]
aux <- unlist(strsplit(TopT_a_s$GPos, ":"))[c(F,T)]
st <- unlist(strsplit(aux, "-"))[c(T,F)]
nd <- unlist(strsplit(aux, "-"))[c(F,T)]


igv <- data.frame(ID = "log10_Pvale_plus_direction_NvsT",
                  chrom = chr,	
                  loc.start = st,
                  loc.end = nd,	
                  num.mark = 1:nrow(TopT_a_s),
                  seg.mean = -log10(TopT_a_s$P.Value) * sign(TopT_a_s$logFC))
write.table(igv, file = sprintf("%s/%s/splicingEvents/IGVtrack.seg",dirResults, cancerSite), quote = F, row.names = F, sep = "\t")


# SF Statistics -----------------------------------------------------------

# load( file = "../hnRNP_Huelga_Analysis/SFprediction/data/input/ExS_Genecode24.RData")
# 
# # Select contrast
# 
# hyperM <- data.frame(RBP = colnames(ExS),
#                      nHits = colSums(ExS),
#                      Pvalue_hyp_PSI =NA,
#                      N = NA,
#                      d = NA,
#                      n = NA,
#                      x = NA,
#                      qhyp_0.5 = NA,
#                      Fc = NA,
#                      stringsAsFactors = F)
# 
# 
# nSel <- 1000
# iTopEv <- order(TopT_a_s$P.Value)[1:nSel]
# 
# for(i in 1:ncol(ExS)){
#   hits <- ExS[,i]
#   N <- nrow(ExS)
#   d <- sum(hits)
#   n <- nSel
#   hits2 <- hits==1
#   #x <- sum(rownames(ExS)[hits] %in% rownames(ExS)[iTopEv])
#   ## rownames(ExS)[hits] da error. Hay que ponerlo en formato TRUE/FALSE
#   x <- sum(rownames(ExS)[hits2] %in% rownames(ExS)[iTopEv])
#   hyperM[i, "Pvalue_hyp_PSI"] <- phyper(x, d, N-d, n, lower.tail = F)
#   qhyp <- qhyper(0.5, d, N-d, n, lower.tail = F)
#   # if(d > 0) {wx <- wilcox.test(Events_F[, iPv] ~ hits)$p.value
#   # }else{wx <- NA}
#   hyperM[i,4:9] <- c(N, d, n, x, qhyp, x/qhyp)
# }
# 
# hyperM <- hyperM[order(hyperM$Pvalue_hyp_PSI), ]
# 
# 
# min(hyperM$Pvalue_hyp_PSI)
# hist(hyperM$Pvalue_hyp_PSI, 100)
# 
# cat(hyperM$RBP[hyperM$Pvalue_hyp_PSI < 0.05], sep = "\n")
# 
# 
