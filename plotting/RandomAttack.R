# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()

no_defense = read.table("./save/RandomAttack/PoisonModel_NoDefense_mnist_cnn_seed1.txt", quote="\"")
norm_bound = read.table("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed1_clip3.2_scale0.0.txt", quote="\"")
GDP_c3.2_s0.01 = read.table("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed1_clip3.2_scale0.01.txt", quote="\"")

for (seed in 2:5){
      no_defense = no_defense + read.table(paste0("./save/RandomAttack/PoisonModel_NoDefense_mnist_cnn_seed", seed, ".txt"), quote="\"")
      norm_bound = norm_bound + read.table(paste0("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed", seed, "_clip3.2_scale0.0.txt"), quote="\"")
      GDP_c3.2_s0.01 = GDP_c3.2_s0.01 + read.table(paste0("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed", seed, "_clip3.2_scale0.01.txt"), quote="\"")
}

no_defense = no_defense / 5
norm_bound = norm_bound / 5
GDP_c3.2_s0.01 = GDP_c3.2_s0.01 / 5


jpeg("./plotting/RandomAttack.jpg")

plot(0:20, no_defense$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, norm_bound$V1, col="red")
lines(0:20, GDP_c3.2_s0.01$V1, col="green")
title(main="1 random poisoning attack gradient each round")
legend("bottomright", c("None", "Norm Bounding", "GDP"), col = c("black", "red", "green"), lty=c(1,1), title = "Defense")

dev.off()