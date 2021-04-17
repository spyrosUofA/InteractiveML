# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()

non_private = read.table("./save/NoAttacks/PrivateFL_mnist_cnn_seed0.txt", quote="\"")
ldp_c3.2_s0.5 = read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.5_seed0.txt", quote="\"")
ldp_c1.6_s0.25 = read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.25_seed0.txt", quote="\"")
gdp_c3.2_s0.05 = read.table("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.05_seed0.txt", quote="\"")
gdp_c2.2_s0.1 = read.table("./save/NoAttacks/GDP_mnist_cnn_norm2.2_scale0.1_seed0.txt", quote="\"")


#GDP_c3.2_s0.01 = read.table("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed1_clip3.2_scale0.01.txt", quote="\"")

for (seed in 1:4){
      non_private = non_private + read.table(paste0("./save/NoAttacks/PrivateFL_mnist_cnn_seed", seed, ".txt"), quote="\"")
      ldp_c3.2_s0.5 = ldp_c3.2_s0.5 + read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.5_seed", seed, ".txt"), quote="\"")
      ldp_c1.6_s0.25 = ldp_c1.6_s0.25 + read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.25_seed", seed, ".txt"), quote="\"")
      gdp_c3.2_s0.05 = gdp_c3.2_s0.05 + read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.05_seed", seed, ".txt"), quote="\"")
      gdp_c2.2_s0.1 = gdp_c2.2_s0.1 + read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm2.2_scale0.1_seed", seed, ".txt"), quote="\"")

      #GDP_c3.2_s0.01 = GDP_c3.2_s0.01 + read.table(paste0("./save/RandomAttack/PoisonModel_GDP_mnist_cnn_seed", seed, "_clip3.2_scale0.01.txt"), quote="\"")
}

# Average across the 5 runs
non_private = non_private / 5
ldp_c3.2_s0.5 = ldp_c3.2_s0.5 / 5
ldp_c1.6_s0.25 = ldp_c1.6_s0.25 / 5
gdp_c3.2_s0.05 = gdp_c3.2_s0.05 /5
gdp_c2.2_s0.1 = gdp_c2.2_s0.1 / 5

jpeg("./plotting/NoAttack.jpg")

plot(0:20, non_private$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, ldp_c3.2_s0.5$V1, col="red")
lines(0:20, ldp_c1.6_s0.25$V1, col="red", lty=2)
lines(0:20, gdp_c3.2_s0.05$V1, col="green")
lines(0:20, gdp_c2.2_s0.1$V1, col="green", lty=2)

title(main="Federated Learning")
legend("bottomright", c("Non-Private", "LDP(e1,d1)", "LDP(e2,d2)", "GDP1", "GDP2"),
        col = c("black", "red", "red", "green", "green"), lty=c(1,1,2,1,2))

dev.off()