# Generate plot of nitrogenase POVME pocket volume data by nitrogenase extant or ancestral sequence
# Amanda Garcia (akgarcia@email.arizona.edu)
# Created: 2019-01-18 Updated: 2019-07-16


# Clear environment
rm(list = ls())


# Read nitrogenase POVME pocket volume data
POVME <- read.csv(file = "Data/PocketVolume_Data.csv") # Edit file path as needed


# Set up plot environment
Seq_plotorder <- scale_x_discrete(limits = c("AncA-1", "AncA-2", "AncA-3", "AncA-4", "AncA-5",
                                             "AncB-1", "AncB-2", "AncB-3", "AncB-4", "AncB-5",
                                             "AncC-1", "AncC-2", "AncC-3", "AncC-4", "AncC-5",
                                             "AncD-1", "AncD-2", "AncD-3", "AncD-4", "AncD-5",
                                             "AncE-1", "AncE-2", "AncE-3", "AncE-4", "AncE-5",
                                             "Nif_Chloroflexi_Kir15-3F", "Nif_Oscillochloris_trichoides", "Nif_Roseiflexus_castenholzii",
                                             "Nif_Caldi_saccharolyticus", "Nif_Meth_thermolithotrophicus", "Nif_Methanocaldo_infernus", "Nif_Moorella_thermoacetica", "Nif_Syntrophotherm_lipocalidus",
                                             "Nif_Methanobacteriales_HGW-1", "Nif_Methanobacterium_paludis", "Nif_Methanobrevi_cuticularis", "Nif_Methanococcus_vannielii_SB", "Nif_Methanothermo_marburgensis",
                                             "Anf_Azotobacter_vinelandii_DJ", "Anf_Brenneria_salicis", "Anf_Clostridium_pasteurianum", "Anf_Dysgon_capnocytophagoides", "Anf_Rhodoblastus_acidophilus",
                                             "Vnf_Azotobacter_vinelandii_DJ", "Vnf_Clostridium_kluyveri", "Vnf_Methanosarcina_acetivorans", "Vnf_Phaeospirillum_fulvum", "Vnf_Rhodoblastus_acidophilus",
                                             "Nif_Chloroherpeton_thalassium", "Nif_Clostridium_kluyveri", "Nif_Dysgon_capnocytophagoides", "Nif_Methanosarcina_acetivorans", "Nif_Methanothrix_soehngenii",
                                             "Nif_Azotobacter_vinelandii_DJ", "Nif_Azovibrio_restrictus", "Nif_Brenneria_salicis", "Nif_Oscillatoria_PCC_10802", "Nif_Rhodoblastus_acidophilus"))

Seq_plotcolor <- scale_fill_manual(values = c("#E0674F", "#E0674F", "#E0674F", "#E0674F", "#E0674F",
                                              "#B779DB", "#B779DB", "#B779DB", "#B779DB", "#B779DB",
                                              "#28A08C", "#28A08C", "#28A08C", "#28A08C", "#28A08C",
                                              "#E8C34B", "#E8C34B", "#E8C34B", "#E8C34B", "#E8C34B",
                                              "#769ECE", "#769ECE", "#769ECE", "#769ECE", "#769ECE",
                                              "#E0674F", "#E0674F", "#E0674F", "#E0674F", "#E0674F",
                                              "#769ECE", "#769ECE", "#769ECE", "#28A08C", "#E8C34B",
                                              "#769ECE", "#769ECE", "#769ECE", "#28A08C", "#B779DB",
                                              "#B779DB", "#B779DB", "#28A08C", "#B779DB", "#769ECE",
                                              "#B779DB", "#769ECE", "#28A08C", "#769ECE", "#E8C34B",
                                              "#769ECE", "#E8C34B", "#28A08C", "#E0674F", "#E0674F",
                                              "#E0674F", "#E0674F", "#E0674F"))


# Boxplot ggplot of pocket volume data, separated by sequence (modeled with FeMo-cofactor)
library(ggplot2)
POVME_Mo_SeqPlot <- ggplot(data = POVME[POVME$Model_cofactor == "Mo", ], 
                        aes(x = Sequence, y = Pocket_volume, fill = Sequence)) + 
  geom_boxplot(outlier.shape = 4, outlier.size = 2) +
  Seq_plotorder +
  Seq_plotcolor + 
  stat_summary(fun.y = "mean", geom = "point") +
  ggtitle("Pocket volume by nitrogenase sequence (Mo)") +
  xlab("Nitrogenase sequence") +
  ylab("Pocket volume (A3)") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "none",
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
POVME_Mo_SeqPlot

# ggsave(filename = "output/POVME_Mo_SeqPlot.svg", plot = POVME_Mo_SeqPlot) # Edit file path as needed; remove # to save


# Boxplot ggplot of pocket volume data, separated by sequence (modeled with FeV-cofactor)
library(ggplot2)
POVME_V_SeqPlot <- ggplot(data = POVME[POVME$Model_cofactor == "V", ], 
                           aes(x = Sequence, y = Pocket_volume, fill = Sequence)) + 
  geom_boxplot(outlier.shape = 4, outlier.size = 2) +
  Seq_plotorder +
  Seq_plotcolor + 
  stat_summary(fun.y = "mean", geom = "point") +
  ggtitle("Pocket volume by nitrogenase group (V)") +
  xlab("Nitrogenase sequence") +
  ylab("Pocket volume (A3)") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "none",
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
POVME_V_SeqPlot

# ggsave(filename = "output/POVME_V_SeqPlot.svg", plot = POVME_V_SeqPlot) # Edit file path as needed; remove # to save