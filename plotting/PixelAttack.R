# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()


no_defenseTA = read.table("./save/PixelAttack/TestAcc/NoDefense_mnist_cnn_attackers2_seed0.txt", quote="\"")
no_defenseBA = read.table("./save/PixelAttack/BackdoorAcc/NoDefense_mnist_cnn_attackers2_seed0.txt", quote="\"")

CDP_TA = read.table("./save/PixelAttack/TestAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed0.txt", quote="\"")
CDP_BA = read.table("./save/PixelAttack/BackdoorAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed0.txt", quote="\"")

for (seed in 1:4){

      # No Defense
      no_defenseTA = no_defenseTA + read.table(paste0("./save/PixelAttack/TestAcc/NoDefense_mnist_cnn_attackers2_seed", seed, ".txt"), quote="\"")
      no_defenseBA = no_defenseBA + read.table(paste0("./save/PixelAttack/BackdoorAcc/NoDefense_mnist_cnn_attackers2_seed", seed, ".txt"), quote="\"")

      # CDP
      CDP_TA = CDP_TA + read.table(paste0("./save/PixelAttack/TestAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed", seed, ".txt"), quote="\"")
      CDP_BA = CDP_BA + read.table(paste0("./save/PixelAttack/BackdoorAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed", seed, ".txt"), quote="\"")


}

no_defenseTA = no_defenseTA / 5
no_defenseBA = no_defenseBA / 5

CDP_TA = CDP_TA /5
CDP_BA = CDP_BA /5

# PLOT TEST ACCURACY
jpeg("./plotting/PixelAttack_TABA.jpg")
par(mfrow=c(1,2))    # set the plotting area into a 1*2 array


plot(0:20, no_defenseTA$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, CDP_TA$V1, col="green")
#lines()
title(main="Test Accuracy")
legend("bottomright", c("None", "LDP", "CDP"), col = c("black", "red", "green"), lty=c(1,1,1), title = "Defense")
#dev.off()



# PLOT BACKDOOR ACCURACY
#jpeg("./plotting/PixelAttack_BA.jpg")
plot(0:20, no_defenseBA$V1, xlab = "Communication Round", ylab = "Backdoor Accuracy", ylim = c(0,1), type="l")
lines(0:20, CDP_BA$V1, col="green")
#lines()
title(main="Backdoor Accuracy")
legend("bottomright", c("None", "LDP", "CDP"), col = c("black", "red", "green"), lty=c(1,1), title = "Defense")
dev.off()
