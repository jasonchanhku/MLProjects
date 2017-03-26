# Chest Clinic Example, Bayes Net

#Libraries Used
library(gRain)
library(gRbase)
library(ggm)


#Building our DAG
g <- list(~asia, ~tub | asia , ~smoke, ~lung | smoke, ~bronc | smoke, ~either | tub:lung , ~xray | either, ~dysp | either:bronc)

chestdag <- dagList(g)

plot(chestdag)

#Query for conditional independence 
dSep(as(chestdag, "matrix"), "bronc", "lung", "smoke")

#Building Bayesian Networks
#Levels
yn <- c("yes", "no")

#Order of values are important 
a <- cptable(~asia, values = c(1,99), levels = yn)

# c(5,95, 1,99) -> p(t|a) , p(~t|a) , p(t|a~) , p(t~|a~)
t.a <- cptable(~tub|asia , values = c(5,95,1,99), levels = yn)
s <- cptable(~smoke, values = c(5,5), levels = yn)
l.s <- cptable(~lung | smoke, values = c(1,9,1,99), levels = yn )
b.s <- cptable(~bronc | smoke, values = c(6,4,3,7), levels = yn)
e.lt <- cptable(~either | lung : tub, values = c(1,0,1,0,1,0,0,1), levels = yn)
x.e <- cptable(~xray | either, values = c(98,2,5,95), levels = yn)
d.be <- cptable(~dysp | bronc : either, values = c(9,1,7,3,8,2,1,9), levels = yn)

#or table for logical
e.lt <- ortable(~either | lung : tub , levels = yn)


#Creating the Network using compilePCT and grain (builds graphical independence network)
plist <- compileCPT(list(a, t.a, s, l.s, b.s, e.lt, x.e, d.be))
grn1 <- grain(plist)
plot(grn1)
summary(grn1)

#Grain object must be compiled and propagated before queries can be made
grn1c <- compile(grn1)
summary(grn1c)

#Check for Cliques
maxCliqueMAT(as(chestdag, "matrix"))

#Triangulated graphs admit closed form MLE and allows simplification by decomposition. A graph is decomposable 
# i.f.f it is triangulated

#Manual compile
g <- grn1$dag
mg <- moralize(g)
tmg <- triangulate(mg)

#Check cliques, separators, and parents
rip(tmg)

#Junction tree, tree that shows organized cliques or a graph i.f.f it is triangulated
plot(grn1c, type = "jt")

#To answer queries, grain object must be propagated 
grn1c <- propagate(grn1c)
summary(grn1c)


#Absorbing evidence and answering queries
#Evidence is first entered using setFinding()
grn1c.ev <- setFinding(grn1c, nodes = c("asia", "dysp"), states = c("yes", "yes"))

# grain objects with and without evidence can be queried to give marginal probabilities using querygrain()
querygrain(grn1c.ev, nodes = c("lung", "bronc"), type = "marginal")

#grain object without evidence
querygrain(grn1c, nodes = c("lung", "bronc"), type = "marginal")

#retrive evidence with the grain object
getFinding(grn1c.ev)

pFinding(grn1c.ev)

#joint and conditional type
querygrain(grn1c.ev, nodes = c("lung", "bronc"), type = "joint")

querygrain(grn1c.ev, nodes = c("lung", "bronc"), type = "conditional")




#Set Findings (evidence) right away will cause grain object to be compiled and propagated
bnet.f <- setFinding(grn1, nodes = c("asia", "dysp"), states = c("yes", "yes"))
bnet.f


#Queries given that A = yes and D = yes, how does it change T, L , and B ? By default, gives marginal distribution of each node
querygrain(bnet.f, nodes = c("lung", "tub", "bronc"), type = "marginal")

#Queries the joint distribution of P(Lung n Tub | Bronc = yes) and P(Lung n Tub | Bronc = no), w.r.t our findings
querygrain(bnet.f, nodes = c("lung", "tub", "bronc"), type = "joint")

#queries the conditional distribution of P(tub | lung)
querygrain(bnet.f, nodes = c("tub", "lung"), type = "conditional")


#probability of p(lung | asia = yes n smoke = no)
bnet.g <- setFinding(grn1, nodes = c("asia", "smoke"), states = c("yes", "no"))
querygrain(bnet.g, nodes = c("lung"), type = "marginal")


#Doing the bnlearn questions

## Question 1
# Patient has recently visited Asia and does not smoke. Which is most
# likely?
# (a) the patient is more likely to have tuberculosis then anything else.
# (b) the chance that the patient has lung cancer is higher than he/she 
#     having tuberculosis
# (c) the patient is more likely to have bronchitis then anything else
# (d) the chance that the patient has tuberculosis is higher than he/she 
#     having bronchitis

#Answer is (c)

bnet.i <- setFinding(grn1, nodes = c("asia", "smoke"), states = c("yes", "no"))
querygrain(bnet.i, nodes = c("tub", "lung", "bronc"), type = "marginal" )



## Question 2
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

#Answer is (b)

bnet.h <- setFinding(grn1, nodes = c("asia", "smoke", "dysp", "xray"), states = c("yes", "no", "no", "yes") )
querygrain(bnet.h, nodes = c("tub", "lung", "bronc"), type = "marginal")



