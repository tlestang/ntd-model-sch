source("analyse_simulations.R")
IUs <- read.csv("mansoniIUs.csv")

IU_data <- read.csv("mansoniIUs_scenario3.csv")

colours <- c("#0098FF", "#1b9e77", "#d95f02","#7570b3", '#d4d133')


path_header <- "~/"
date_name <- "20220822a"

### one IU
scenario <- "0"
IU <- "AGO02069"

ipm_name <- write_file_name(IU, scenario, path_header, date_name, IU_data,
                            output = "ipm")

res <- read_and_get_number_mda_finished(IU, ipm_name, IU_data, interval = TRUE,
                                        single_IU = TRUE)
years <- 2018:2041
plot(years, res$means, type = 'l', lwd = 4, 
     xlab = 'year', ylab = 'proportion IUs stop MDA',
     col = colours[1],
     bty = 'n', 
     ylim = c(0,max(res$interval)),
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,cex.main = 1.7)
lines(years, res$interval[1,], col = colours[1], lty = 2)
lines(years, res$interval[2,], col = colours[1], lty = 2)

### loop over scenarios
IU <- "AGO02069"
years <- 2018:2041
plot(NA, xlim = c(min(years), max(years)), ylim = c(0, 0.2),
     xlab = 'year', ylab = 'proportion IUs stop MDA',
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,cex.main = 1.7)

par(mfrow = c(3,2))
scenario_vec <- c("0", "1", "2", "3a", "3b")
for(i in 1:length(scenario_vec)){
  scenario <- scenario_vec[i]
  ipm_name <- write_file_name(IU, scenario, path_header, date_name, IU_data,
                               output="ipm")
  res <- read_and_get_number_mda_finished(IU, ipm_name, IU_data, interval = TRUE,
                                          single_IU = TRUE)
  
  plot(years, res$means, type = 'l', col = colours[i])
  lines(years, res$interval[1,], col = colours[i], lty = 2)
  lines(years, res$interval[2,], col = colours[i], lty = 2)
}


### loop over multiple IUs
IU_data <- read.csv("mansoniIUs_scenario3.csv")
IU_data <- IU_data[1:20,]

scenario_1_res <- get_number_mda_stopped_multiple_IUS(scenario = "1", IU_data, path_header, date_name)
num_mda_finished_1_1 <- scenario_1_res$num_finished_1
num_mda_finished_1_2 <- scenario_1_res$num_finished_2
totalIUS_1 <- scenario_1_res$totalIUS_1
totalIUS_2 <- scenario_1_res$totalIUS_2

###loop over all Ius and all scenarios
scenario_0_res <- get_number_mda_stopped_multiple_IUS(scenario = "0", IU_data, path_header, date_name)
num_mda_finished_0_1 <- scenario_0_res$num_finished_1
num_mda_finished_0_2 <- scenario_0_res$num_finished_2


scenario_2_res <- get_number_mda_stopped_multiple_IUS(scenario = "2", IU_data, path_header, date_name)
num_mda_finished_2_1 <- scenario_2_res$num_finished_1
num_mda_finished_2_2 <- scenario_2_res$num_finished_2


scenario_3a_res <- get_number_mda_stopped_multiple_IUS(scenario = "3a", IU_data, path_header, date_name)
num_mda_finished_3a_1 <- scenario_3a_res$num_finished_1
num_mda_finished_3a_2 <- scenario_3a_res$num_finished_2


scenario_3b_res <- get_number_mda_stopped_multiple_IUS(scenario = "3b", IU_data, path_header, date_name)
num_mda_finished_3b_1 <- scenario_3b_res$num_finished_1
num_mda_finished_3b_2 <- scenario_3b_res$num_finished_2



cols = c("#0098FF", "#1b9e77", "#d95f02","#7570b3", '#d4d133')
prop_finished_0_1 = num_mda_finished_0_1/totalIUS_1 
prop_finished_1_1 = num_mda_finished_1_1/totalIUS_1 
prop_finished_2_1 = num_mda_finished_2_1/totalIUS_1 
prop_finished_3a_1 = num_mda_finished_3a_1/totalIUS_1 
prop_finished_3b_1 = num_mda_finished_3b_1/totalIUS_1 
png("Mansoni_group1_mda_stopping.png", units = "in",
    width = 12, height = 8, res = 300)
plot(2018:2041, prop_finished_0_1 , type = 'l', lwd = 4, 
     xlab = 'year', ylab = 'proportion IUs stopped MDA',
     col = 0,
     bty = 'n', ylim = c(0, 1),
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,
     cex.main = 1.7, main = "Mansoni Group 1")
lines(2018:2041, prop_finished_1_1 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[2],
      bty = 'n')
lines(2018:2041, prop_finished_2_1 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[3],
      bty = 'n')

lines(2018:2041, prop_finished_3a_1 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[4],
      bty = 'n')

lines(2018:2041, prop_finished_3b_1 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[5],
      bty = 'n')

# legend('bottomright',  legend = c("Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
#        col = cols, lwd = c(4,4,4,4,4),
#        bty = 'n', cex = 1.7)


legend('bottomright',  legend = c("Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
       col = cols[-1], lwd = c(4,4,4,4),
       bty = 'n', cex = 1.7)
dev.off()


prop_finished_0_2 = num_mda_finished_0_2/totalIUS_2
prop_finished_1_2 = num_mda_finished_1_2/totalIUS_2 
prop_finished_2_2 = num_mda_finished_2_2/totalIUS_2 
prop_finished_3a_2 = num_mda_finished_3a_2/totalIUS_2 
prop_finished_3b_2 = num_mda_finished_3b_2/totalIUS_2 
png("Mansoni_group2_mda_stopping.png", units = "in",
    width = 12, height = 8, res = 300)
plot(2018:2041, prop_finished_0_2 , type = 'l', lwd = 4, 
     xlab = 'year', ylab = 'proportion IUs stopped MDA',
     col = 0,
     bty = 'n', ylim = c(0, 1),
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,
     cex.main = 1.7, main = "Mansoni Group 2")

lines(2018:2041, prop_finished_1_2 , type = 'l', lwd = 4, 
      
      col = cols[2],
      bty = 'n')

lines(2018:2041, prop_finished_2_2 , type = 'l', lwd = 4, 
      
      col = cols[3],
      bty = 'n')

lines(2018:2041, prop_finished_3a_2 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[4],
      bty = 'n')

lines(2018:2041, prop_finished_3b_2 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[5],
      bty = 'n')

# legend('bottomright',  legend = c("Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
#        col = cols, lwd = c(4,4,4,4,4),
#        bty = 'n', cex = 1.7)

legend('bottomright',  legend = c("Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
       col = cols[-1], lwd = c(4,4,4,4),
       bty = 'n', cex = 1.7)

dev.off()


prop_finished_0 = (num_mda_finished_0_1 + num_mda_finished_0_2)/(totalIUS_1 +totalIUS_2)
prop_finished_1 = (num_mda_finished_1_1 + num_mda_finished_1_2)/(totalIUS_1 +totalIUS_2)
prop_finished_2 = (num_mda_finished_2_1 + num_mda_finished_2_2)/(totalIUS_1 +totalIUS_2)
prop_finished_3a = (num_mda_finished_3a_1 + num_mda_finished_3a_2)/(totalIUS_1 +totalIUS_2)
prop_finished_3b = (num_mda_finished_3b_1 + num_mda_finished_3b_2)/(totalIUS_1 +totalIUS_2)

png("Mansoni_mda_stopping.png", units = "in",
    width = 12, height = 8, res = 300)
plot(2018:2041, prop_finished_0 , type = 'l', lwd = 4, 
     xlab = 'year', ylab = 'proportion IUs stopped MDA',
     col = 0,
     bty = 'n', ylim = c(0, 1),
     cex = 1.7, cex.axis = 1.7, cex.lab = 1.7,
     cex.main = 1.7, main = "Mansoni")

lines(2018:2041, prop_finished_1 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[2],
      bty = 'n')

lines(2018:2041, prop_finished_2 , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[3],
      bty = 'n')

lines(2018:2041, prop_finished_3a , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[4],
      bty = 'n')

lines(2018:2041, prop_finished_3b , type = 'l', lwd = 4, 
      xlab = 'year', ylab = 'proportion IUs stop MDA',
      col = cols[5],
      bty = 'n')

# legend('bottomright',  legend = c("Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
#        col = cols, lwd = c(4,4,4,4,4),
#        bty = 'n', cex = 1.7)

legend('bottomright',  legend = c("Scenario 1", "Scenario 2", "Scenario 3a","Scenario 3b"),
       col = cols[-1], lwd = c(4,4,4,4),
       bty = 'n', cex = 1.7)


dev.off()
