# Title     : TODO
# Objective : TODO
# Created by: spyros
# Created on: 2021-04-14
getwd()

no_defense = read.table("./save/PoisonModel_NoDefense_mnist_cnn_seed1.txt", quote="\"")
GDP_c3.2_s0 = read.table("./save/PoisonModel_GDP_mnist_cnn_seed1_clip3.2_scale0.0.txt", quote="\"")

for (seed in 2:5){
      no_defense = no_defense + read.table(paste0("./save/PoisonModel_NoDefense_mnist_cnn_seed", seed, ".txt"), quote="\"")
      GDP_c3.2_s0 = GDP_c3.2_s0 + read.table(paste0("./save/PoisonModel_GDP_mnist_cnn_seed", seed, "_clip3.2_scale0.0.txt"), quote="\"")
}

no_defense = no_defense / 5
GDP_c3.2_s0 = GDP_c3.2_s0 / 5

plot(0:20, no_defense$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, GDP_c3.2_s0$V1, col="red")