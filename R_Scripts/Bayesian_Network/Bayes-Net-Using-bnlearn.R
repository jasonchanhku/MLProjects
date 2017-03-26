library(bnlearn)

#IMPORTANT: bnlearn uses I'm not very satisfied with its inference engine. It uses sampling methods, so two 
#consecutive calls to cpquery with the same arguments will likely return (more or less) slightly different results.
#best way to handle this is to average the probability, by LLN, it should approach the theoretical probability


prob1 <- c()
prob2 <- c()
prob3 <- c()

set.seed(42)

#Building the BN
net <- model2network("[A] [S] [T|A] [L|S] [B|S] [E|T:L] [X|E] [D|B:E]")
yn <- c("yes", "no")

#Use matrix to create the nodes
cptA <- matrix(c(0.01, 0.99), ncol = 2, dimnames = list(NULL, yn))
cptS <- matrix(c(0.5, 0.5), ncol=2, dimnames=list(NULL, yn))
matrix(c(0.05, 0.95, 0.01, 0.99), ncol = 2, dimnames = list("T" = yn, "A" = yn))
cptT <- matrix(c(0.05, 0.95, 0.01, 0.99), 
               ncol=2, dimnames=list("T"=yn, "A"=yn))
cptL <- matrix(c(0.1, 0.9, 0.01, 0.99), 
               ncol=2, dimnames=list("L"=yn, "S"=yn))
cptB <- matrix(c(0.6, 0.4, 0.3, 0.7), 
               ncol=2, dimnames=list("B"=yn, "S"=yn))

# cptE and cptD are 3-d matrices, which don't exist in R, so
# need to build these manually as below.
cptE <- c(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
dim(cptE) <- c(2, 2, 2)
dimnames(cptE) <- list("E"=yn, "L"=yn, "T"=yn)
cptX <- matrix(c(0.98, 0.02, 0.05, 0.95), 
               ncol=2, dimnames=list("X"=yn, "E"=yn))
cptD <- c(0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.1, 0.9)
dim(cptD) <- c(2, 2, 2)
dimnames(cptD) <- list("D"=yn, "E"=yn, "B"=yn)
net.disc <- custom.fit(net, dist=list(A=cptA, S=cptS, T=cptT, L=cptL, 
                                      B=cptB, E=cptE, X=cptX, D=cptD))

#Plot
graphviz.plot(net.disc)

#Given no evidence (findings), what are the chances of Tuberculosis, Lung Cancer, and Bronchitis ?
# cpquery(fitted, event, evidence, cluster = NULL, method = "ls", ..., debug = FALSE)
?cpquery

# P(T)
cpquery(net.disc, (T == "yes"), TRUE)

# P(L)
cpquery(net.disc, (L == "yes"), TRUE)

# P(B)
cpquery(net.disc, (B == "yes"), TRUE)


# Question 1:
# Patient has recently visited Asia and does not smoke. Which is most
# likely?
# (a) the patient is more likely to have tuberculosis then anything else.
# (b) the chance that the patient has lung cancer is higher than he/she 
#     having tuberculosis
# (c) the patient is more likely to have bronchitis then anything else
# (d) the chance that the patient has tuberculosis is higher than he/she 
#     having bronchitis

#Evidence -> A = yes , S = no. 
# Probabilities 0.30053830 0.05077798 0.01012742. Hence the answer is (c)
for( i in 1:2000)
{
prob1[i] <- cpquery(net.disc, (B=="yes"), (A=="yes" & S == "no"))
prob2[i] <- cpquery(net.disc, (T=="yes"), (A=="yes" & S=="no"))
prob3[i] <- cpquery(net.disc, (L=="yes"), (A=="yes" & S=="no"))
}

list( Probabilities = c(mean(prob1), mean(prob2), mean(prob3)), Stdev = c(sqrt(var(prob1)), sqrt(var(prob2)), sqrt(var(prob3))))


# Question 2
# The patient has recently visited Asia, does not smoke, is not 
# complaining of dyspnoea, but his/her x-ray shows a positive shadow
# (a) the patient most likely has tuberculosis, but lung cancer is 
#     almost equally likely
# (b) the patient most likely has tuberculosis as compared to any of 
#     the other choices
# (c) the patient most likely has bronchitis, and tuberculosis is 
#     almost equally likely
# (d) the patient most likely has tuberculosis, but bronchitis is 
#     almost equally likely

#(d) is the answer
for(i in 1:5000){
prob1[i] <- cpquery(net.disc, (T=="yes"), (A=="yes" & S=="no" & D=="no" & X=="yes"))
prob2[i] <- cpquery(net.disc, (L=="yes"), (A=="yes" & S=="no" & D=="no" & X=="yes"))
prob3[i] <- cpquery(net.disc, (B=="yes"), (A=="yes" & S=="no" & D=="no" & X=="yes"))
}

list(Probabilities = c(mean(prob1), mean(prob2), mean(prob3)), Variance = c(var(prob1), var(prob2), var(prob3)))
