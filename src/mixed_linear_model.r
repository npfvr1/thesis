library(lme4)     # For GLMMs
library(dplyr)    # For data manipulation
library(ggplot2)  # For plotting
library(readxl)   # For reading Excel files
library(lmerTest) # p-values (not part of the standard lme4 packages)


# ---- DATA ----
data <- read_excel("data/processed/lmm_data.xlsx")
data$drug <- as.factor(data$drug)
data$time <- as.factor(data$time)


# ---- LINEAR MIXED-EFFECTS MODELS ----
cat("\n\n---------------- DELTA ----------------\n\n")
# Effect of the interaction
model_delta <- lmer(delta ~ drug + time + drug:time + (1 | id), data = data)
# print(summary(model_delta))
model_delta_reduced <- lmer(delta ~ drug + time + (1 | id), data = data)                      
anova_result <- anova(model_delta_reduced, model_delta, test = "LRT")
print(anova_result)

# Effect of time
model_delta <- lmer(delta ~ drug + time + drug:time + (1 | id), data = data)
model_delta_reduced <- lmer(delta ~ drug + (1 | id), data = data)                      
anova_result <- anova(model_delta_reduced, model_delta, test = "LRT")
print(anova_result)

# Effect of drug TODO : do this for each feature and report results
model_delta <- lmer(delta ~ drug + time + drug:time + (1 | id), data = data)
model_delta_reduced <- lmer(delta ~ time + (1 | id), data = data)                      
anova_result <- anova(model_delta_reduced, model_delta, test = "LRT")
print(anova_result)




stop()



cat("\n\n---------------- THETA ----------------\n\n")
model_theta <- lmer(theta ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_theta))

cat("\n\n---------------- ALPHA ----------------\n\n")
model_alpha <- lmer(alpha ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_alpha))

cat("\n\n---------------- RATIO ----------------\n\n")
model_ratio <- lmer(ratio ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_ratio))

cat("\n\n---------------- PE ----------------\n\n")
model_pe <- lmer(pe ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_pe))

cat("\n\n---------------- SE ----------------\n\n")
model_se <- lmer(se ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_se))

cat("\n\n---------------- FNIRS ----------------\n\n")
model_fnirs <- lmer(fnirs_1 ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_fnirs))

cat("\n\n---------------- PUPILLOMETRY ----------------\n\n")
model_pupillometry <- lmer(pupillometry_score ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_pupillometry))


stop()


# Tobias: Fit the model without the effect of drug for comparison
model_behavior_reduced <- glmer(behavior_change ~ 1 + (1 | record_id),
                                data = data3,
                                family = binomial(link = "logit"), nAGQ = 100)

# Tobias: Compare the models as in a main effect ANOVA. Not significant
anova_result <- anova(model_behavior_reduced, model_behavior, test = "LRT")
print(anova_result)



# Calculate odds ratios
#odds_ratios <- exp(coef(model_behavior))
#Tobias: updated for glmer
odds_ratios <- exp(fixef(model_behavior))


# Print the odds ratios
print(odds_ratios)

# Calculate confidence intervals for coefficients
#conf_int <- confint(model_behavior)
#Tobias: updated for glmer
conf_int <- confint(model_behavior, parm = "beta_", level = 0.95)

# Exponentiate to get confidence intervals for odds ratios
#odds_ratio_ci <- exp(conf_int)
#Tobias: updated for glmer
odds_ratio_ci <- exp(conf_int)

# Print the confidence intervals for the odds ratios
print(odds_ratio_ci)

# Combine odds ratios and their confidence intervals into a data frame
results <- data.frame(
  Odds_Ratios = odds_ratios,
  CI_Lower = odds_ratio_ci[, 1],
  CI_Upper = odds_ratio_ci[, 2]
)

# Print the results
print(results)

# Creating a data frame for plotting
drug_effects <- data.frame(
  Drug = c("Placebo", "Methylphenidate", "Apomorphine"),
  Odds_Ratios = c(0.02777778, 7.875, 4.5),
  CI_Lower = c(0.001563616, 1.302339938, 0.625676192),
  CI_Upper = c(0.1281871, 151.4020729, 90.4933902)
)

# Load necessary library
library(ggplot2)
library(scales)

# Create the plot
odds_ratio_plot <- ggplot(drug_effects, aes(x = Drug, y = Odds_Ratios, ymin = CI_Lower, ymax = CI_Upper)) +
  geom_point(size = 4) +
  geom_errorbar(width = 0.2, size = 1) +
  scale_y_log10(breaks = c(0.1, 1, 10, 100), labels = scales::trans_format("log10", math_format(10^.x))) +
  labs(title = "Odds Ratios and 95% Confidence Intervals for Drug Effects on Behavior Change",
       x = "Drug Treatment",
       y = "Odds Ratio (log scale)") +
  theme_minimal()

# Print the plot
print(odds_ratio_plot)
