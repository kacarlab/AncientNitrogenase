# Dunn's test of POVME data by nitrogenase extant or ancestral sequence
# Amanda Garcia (akgarcia@email.arizona.edu)
# Created: 2019-02-22 Updated: 2019-07-16


# Clear environment
rm(list = ls())


# Read nitrogenase POVME pocket volume data
POVME <- read.csv(file = "Data/PocketVolume_Data.csv") # Edit file path as needed


# Parse POVME data modeled with FeMo-cofactor
POVME_Mo <- POVME[grep("Mo", POVME$Model_cofactor), ]


# Parse POVME data modeled with FeV-cofactor
POVME_V <- POVME[grep("V", POVME$Model_cofactor), ]


# Run Kruskall-Wallis test on pocket volume data by sequence (from both  Mo and V models)
# Null hypothesis is that the location parameters of the distribution of x are the same in each group (sample). The alternative is that they differ in at least one.
POVME_Mo_Seq_Kruskal <- kruskal.test(Pocket_volume ~ Sequence, POVME_Mo)
summary(POVME_Mo_Seq_Kruskal)

POVME_V_Seq_Kruskal <- kruskal.test(Pocket_volume ~ Sequence, POVME_V)
summary(POVME_V_Seq_Kruskal)

# Run Dunn's test for pairwise comparison of sequence pocket volume median values
# Null hypothesis for each pairwise comparison is that the probability of observing a
# randomly selected value from the first group that is larger than a randomly selected value from the second group equals one half
library(dunn.test)

POVME_Mo_Seq_Dunn <- dunn.test(x = POVME_Mo$Pocket_volume, g = POVME_Mo$Sequence)
# write.csv(POVME_Mo_Seq_Dunn, file = "Output/POVME_Mo_Seq_Dunn.csv") # Edit file path as needed; remove # to save

POVME_V_Seq_Dunn <- dunn.test(x = POVME_V$Pocket_volume, g = POVME_V$Sequence)
# write.csv(POVME_V_Seq_Dunn, file = "Output/POVME_V_Seq_Dunn.csv") # Edit file path as needed; remove # to save