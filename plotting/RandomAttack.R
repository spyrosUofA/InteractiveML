# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()

no_defense = read.table("./save/RandomAttack/PoisonModel_NoDefense_mnist_cnn_seed1.txt", quote="\"")
GDP = read.table("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed1_clip3.2_scale0.01.txt", quote="\"")
LDP = read.table("./save/RandomAttack/LDP_FL_mnist_cnn_norm3.2_scale0.04_seed0.txt", quote="\"")

for (seed in 2:5){
      no_defense = no_defense + read.table(paste0("./save/RandomAttack/PoisonModel_NoDefense_mnist_cnn_seed", seed, ".txt"), quote="\"")
      GDP = GDP + read.table(paste0("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed", seed, "_clip3.2_scale0.01.txt"), quote="\"")
}


for (seed in 1:4){
      LDP = LDP + read.table(paste0("./save/RandomAttack/LDP_FL_mnist_cnn_norm3.2_scale0.04_seed", seed, ".txt"), quote="\"")
}


no_defense = no_defense / 5
GDP = GDP / 5
LDP = LDP / 5


jpeg("./plotting/RandomAttack.jpg")

plot(0:20, no_defense$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, LDP$V1, col="red")
lines(0:20, GDP$V1, col="green")
title(main="1 random poisoning attack gradient each round")
legend("bottomright", c("None", "LDP", "GDP"), col = c("black", "red", "green"), lty=c(1,1), title = "Defense")

dev.off()