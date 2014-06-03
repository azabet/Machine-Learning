#####################################################################
## Amir Zabet @ 06/02/2014
## Machine Learning Project
## Spam Filtering using Naive Bayes
## Reference: "Doing Data Science"
##	-> http://shop.oreilly.com/product/0636920028529.do
## Data: Enron emails 
##	-> http://www.csmining.org/index.php/enron-spam-datasets.html
## Data Mining using tm (textmining) package
## 	-> http://cran.r-project.org/web/packages/tm/index.html
#####################################################################


rm(list=ls(all=TRUE))	## Clear the memory
library(tm) 		## load the tm package


#####################################################################
## Function: read.files(label)
## Create a 'spam' or 'ham' corpus using the tm package
## Save the result in .RData file
#####################################################################
read.files <- function(label) {
	if (!label %in% c("spam","ham")) stop("label must be 'ham' or 'spam'")
	path <- paste0("./",label,"/")
	## Create corpus
	myCorpus <- Corpus(DirSource(path))
	## Convert to lower case
	myCorpus <- tm_map(myCorpus, tolower)
	## Remove punctuation
	myCorpus <- tm_map(myCorpus, removePunctuation)
	## Remove stopwords
	myCorpus2 <- tm_map(myCorpus, removeWords, stopwords('english'))
	## Remove numbers
	myCorpus <- tm_map(myCorpus, removeNumbers)
	## Save the result
	save(myCorpus, file=paste0(label,".corpus.RData"))
}


#####################################################################
## Function train.test(training)
## Split the spam and ham corpora into training and test sets
## Store the words in TextDocumentMatrix
## Save the result in train.test.RData 
#####################################################################
train.test <- function(training=0.9){
	if (training>=1 | training<=0) stop("training must be between 0 and 1")
	## Randomly split the spam emails
	load(paste0("spam.corpus.RData"))
	N <- length(myCorpus)
	Ntest <- N*(1-training)
	test.set <- sample(1:N, Ntest, replace=F)
	training.set <- setdiff(1:N, test.set)
	spam.test <- myCorpus[test.set]
	spam.training <- myCorpus[training.set]
	## Create TermDocumentMatrix for spam training
	spam.TDM <- TermDocumentMatrix(spam.training, control=(list(wordLengths=c(1,Inf))))
	## Randomly split the ham emails
	load(paste0("ham.corpus.RData"))
	N <- length(myCorpus)
	Ntest <- N*(1-training)
	test.set <- sample(1:N, Ntest, replace=F)
	training.set <- setdiff(1:N, test.set)
	ham.test <- myCorpus[test.set]
	ham.training <- myCorpus[training.set]
	## Create TermDocumentMatrix for ham training
	ham.TDM <- TermDocumentMatrix(ham.training, control=(list(wordLengths=c(1,Inf))))
	## Create TermDocumentMatrix for test
	test.TDM <- TermDocumentMatrix(c(spam.test,ham.test), control=(list(wordLengths=c(1,Inf))))
	## Store the true labels (1 = spam; 0 = ham)
	test.true.labels <- c(rep(1,length(spam.test)), rep(0,length(ham.test)))
	## Save the result
	save(spam.training, spam.test, spam.TDM,
		ham.training, ham.test, ham.TDM,
		test.TDM, test.true.labels, file="train.test.RData")
}


#####################################################################
## Function word.freq(TDM, sortBy='freq')
## Calculate word frequencies from a TextDocumentMatrix (TDM)
## Result can be sorted either alphabetically or
##	by the decreasing order of frequencies (default)
#####################################################################
word.freq <- function(TDM, sortBy='freq'){
	## TDM is a 'simple_triplet_matrix' object (package: slam)
	## Use 'row_sums' to calculate frequencies across the rows
	freq <- slam::row_sums(TDM)
	if (sortBy=='freq') freq <- sort(freq,decreasing=T)
	return(freq)
}


#####################################################################
## Main program
#####################################################################
## Read the data
if (!file.exists("spam.corpus.RData")) read.files("spam")
if (!file.exists("ham.corpus.RData")) read.files("ham")

## Create training/test sets
if (!file.exists("train.test.RData")) train.test(0.9)
load("train.test.RData")

## Analyze spam words
spam.TDM <- weightBin(spam.TDM)  ## binary weight; ignore repetitions in a single email
spam.N <- dim(spam.TDM)[2]
spam.word.freq <- word.freq(spam.TDM)
spam.words <- names(spam.word.freq)

## Analyze ham words
ham.TDM <- weightBin(ham.TDM)
ham.N <- dim(ham.TDM)[2]
ham.word.freq <- word.freq(ham.TDM)
ham.words <- names(ham.word.freq)

## Find words that appear only in spam
## Store them in spam.filter1
common <- spam.words %in% ham.words
spam.filter1 <- spam.words[!common]
cat(length(spam.filter1),"words appear in spam only -> added to spam filter\n\n")
cat(sum(common),"words appear in both spam and ham\n")

## Calculate spam probability for the common words
## Naive Bayes:
## p(spam|word) = p(word|spam)*p(spam)/p(word)
## p(word) = p(word|spam)*p(spam) + p(word|ham)*p(ham)
p.spam <- spam.N/(spam.N+ham.N)
p.ham  <- 1 - p.spam
common.words <- spam.words[common]
p.word.spam <- spam.word.freq[common]/spam.N
p.word.ham <- ham.word.freq[common.words]/ham.N
p.word <- p.word.spam*p.spam + p.word.ham*p.ham
p.spam.word <- p.word.spam*p.spam / p.word
spam.prob <- data.frame(p.word,p.spam.word)

## Display the result
cat("\nTraining result (sorted by p.word):\n")
cat("-----------------------------------\n")
sorter <- order(spam.prob$p.word,decreasing=T)
spam.prob <- spam.prob[sorter,]
print(head(spam.prob))
cat("\nTraining result (sorted by p.spam.word):\n")
cat("----------------------------------------\n")
sorter <- order(spam.prob$p.spam.word,decreasing=T)
spam.prob <- spam.prob[sorter,]
print(head(spam.prob))

## Set a spam probability threhold
## Store the words with higher probability in spam.filter2
thld <- 0.90
cat("\nThreshold for labeling spams:",thld,"\n")
spam.filter2 <- names(p.spam.word[p.spam.word>=thld])
cat(length(spam.filter2),"additional words added to spam filter\n")
spam.filter <- c(spam.filter1,spam.filter2)

## Analyze the test set
Ntest <- test.TDM$ncol
terms <- test.TDM$dimnames$Terms
test.spam   <- which(test.true.labels==1)
test.spam.N <- length(test.spam)
test.ham    <- which(test.true.labels==0)
test.ham.N  <- length(test.ham)
prediction <- array()
for (j in 1:Ntest){
	label <- 0	## default: label = ham
	word.list <- terms[test.TDM$i[test.TDM$j==j]]
	if (any(word.list %in% spam.filter))  label <- 1	## label = spam
	prediction[j] <- label
}
## Create confusion table
TP <- sum(prediction[test.spam]==1)/test.spam.N
FN <- sum(prediction[test.spam]==0)/test.spam.N
FP <- sum(prediction[test.ham]==1)/test.ham.N
TN <- sum(prediction[test.ham]==0)/test.ham.N
confusion.table <- rbind(c(TP,FP),c(FN,TN))
rownames(confusion.table) <- c("spam","ham")
colnames(confusion.table) <- c("spam","ham")
names(dimnames(confusion.table)) <- c("Predicted","Actual")
print(confusion.table)
accuracy <- (TP*test.spam.N+ TN*test.ham.N)/(test.spam.N+test.ham.N)
cat("accuracy:",accuracy,"\n")

## Try to improve the accuracy by increasing the number of matched words 
prediction <- array()
for (k in 2:10) {
	for (j in 1:Ntest){
		label <- 0
		word.list <- terms[test.TDM$i[test.TDM$j==j]]
		if (sum(word.list %in% spam.filter)>=k)  label <- 1
		prediction[j] <- label
	}
	TP[k] <- sum(prediction[test.spam]==1)/test.spam.N
	FN <- sum(prediction[test.spam]==0)/test.spam.N
	FP <- sum(prediction[test.ham]==1)/test.ham.N
	TN[k] <- sum(prediction[test.ham]==0)/test.ham.N
	confusion.table <- rbind(c(TP,FP),c(FN,TN))
	rownames(confusion.table) <- c("spam","ham")
	colnames(confusion.table) <- c("spam","ham")
	names(dimnames(confusion.table)) <- c("Predicted","Actual")
	print(confusion.table)
	accuracy[k] <- (TP[k]*test.spam.N+ TN[k]*test.ham.N)/(test.spam.N+test.ham.N)
	cat(k, accuracy[k], "\n")
}
plot(accuracy, type='o', pch=16, xlab='Minimum no. of matched words')
abline(v=which.max(accuracy), lty=3)
## Plot accuracy, TP, and TN
plot(0, 0, xlim=c(1,10), ylim=c(min(TP),max(TN)), type="n", 
     xlab='Minimum no. of matched words', ylab='Probability')
points(accuracy, type='o', pch=16, col='black')
points(TP, type='o', pch=1, col='red')
points(TN, type='o', pch=2, col='blue')
abline(v=which.max(accuracy), lty=3)
legend(7.5,0.75,c("accuracy","true positive","true negative"),col=c("black","red","blue"),pch=c(16,1,2),lty=1)
