# HISTORICAL CODE EXCERPT — NOT FOR EXECUTION OR PUBLICATION ANALYSIS
# Source: recovered Main_Analysis_History_UNVERIFIED.R from dissertation records.
# Retains original lines 1–379 for methodological provenance only.
# Later lines (380 onwards) were deliberately excluded because that section
# includes hard-coded numerical research results/correlation matrices, along
# with incomplete experimental code; those values are not for public release.
# This excerpt has NOT been corrected or validated and contains exploratory
# GLMs/occupancy models that require reassessment for compositional dependence.
# Some objects/functions are undefined; do not source this file end-to-end.
# The complete historical record is retained separately in private materials.

# Specify the number of latent classes
num_classes <- 3
# Run LCA model
lca_model <- Mclust(lca_data, G = num_classes)
# View model results
summary(lca_model)
# Plot item-response probabilities for each latent class
plot(lca_model)
# Load required packages
install.packages("mclust")
library(mclust)
# Load the dataset
my_data <- read.csv("SESWithImputed.csv")
# Function to impute missing values with median for numeric columns and mode for categorical columns
impute_missing <- function(x) {
for (i in 1:ncol(x)) {
if (is.numeric(x[, i])) {
x[is.na(x[, i]), i] <- median(x[, i], na.rm = TRUE)
} else {
levels <- unique(x[, i])
mode <- levels[which.max(tabulate(match(x[, i], levels)))]
x[is.na(x[, i]), i] <- mode
}
}
return(x)
}
# Impute missing values
my_data <- impute_missing(my_data)
# Specify variables for LCA
lca_vars <- c("Equivalised_Income_Updated", "Subjective_SES", "MATRIX_Tscore")
# Subset data to include only the selected variables
lca_data <- my_data[, lca_vars]
# Specify the number of latent classes
num_classes <- 3
# Run LCA model
lca_model <- Mclust(lca_data, G = num_classes)
# View model results
summary(lca_model)
# Plot item-response probabilities for each latent class
plot(lca_model)
ggplot(your_data, aes(x = MATRIX_Tscore, y = SwitchingRate)) +
geom_point(color = "#0ff0fc", size = 3, alpha = 0.7) +  # Change color and transparency of points
geom_smooth(method = "glm", formula = y ~ x, color = "#ff8e4d", linetype = "dashed", size = 1.5, se = TRUE, alpha = 0.5, fill = "#b4fbff") +  # Change line style, color, and transparency of the trend line
labs(
x = "MATRIX_Tscore",
y = "SwitchingRate",
title = "Scatter Plot of SwitchingRate by MATRIX_Tscore",
subtitle = "with Trend Lines and Confidence Intervals"
) +
theme_minimal() +  # Use a minimal theme
theme(
plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Adjust title size and alignment
plot.subtitle = element_text(size = 14, hjust = 0.5),  # Adjust subtitle size and alignment
axis.text = element_text(size = 12),  # Adjust axis text size
axis.title = element_text(size = 14, face = "bold"),  # Adjust axis title size and weight
legend.title = element_blank(),  # Remove legend title
legend.text = element_text(size = 12),  # Adjust legend text size
plot.caption = element_blank()  # Remove caption
)
# Load ggplot2 package
library(ggplot2)
# Plot with enhanced aesthetics
ggplot(SESWithImputed, aes(x = MATRIX_Tscore, y = SwitchingRate)) +
geom_point(color = "#0ff0fc", size = 3, alpha = 0.7) +  # Change color and transparency of points
geom_smooth(method = "glm", formula = y ~ x, color = "#ff8e4d", linetype = "dashed", size = 1.5, se = TRUE, alpha = 0.5, fill = "#b4fbff") +  # Change line style, color, and transparency of the trend line
labs(
x = "MATRIX_Tscore",
y = "SwitchingRate",
title = "Scatter Plot of SwitchingRate by MATRIX_Tscore",
subtitle = "with Trend Lines and Confidence Intervals"
) +
theme_minimal() +  # Use a minimal theme
theme(
plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Adjust title size and alignment
plot.subtitle = element_text(size = 14, hjust = 0.5),  # Adjust subtitle size and alignment
axis.text = element_text(size = 12),  # Adjust axis text size
axis.title = element_text(size = 14, face = "bold"),  # Adjust axis title size and weight
legend.title = element_blank(),  # Remove legend title
legend.text = element_text(size = 12),  # Adjust legend text size
plot.caption = element_blank()  # Remove caption
)
# Load ggplot2 package
library(ggplot2)
# Plot with enhanced aesthetics
ggplot(SESWithImputed, aes(x = SDQ_Hyperactivity, y = SwitchingRate)) +
geom_point(color = "#0ff0fc", size = 3, alpha = 0.7) +  # Change color and transparency of points
geom_smooth(method = "glm", formula = y ~ x, color = "#ff8e4d", linetype = "dashed", size = 1.5, se = TRUE, alpha = 0.5, fill = "#b4fbff") +  # Change line style, color, and transparency of the trend line
labs(
x = "SDQ_Hyperactivity",
y = "SwitchingRate",
title = "Scatter Plot of SwitchingRate by SDQ_Hyperactivity",
subtitle = "with Trend Lines and Confidence Intervals"
) +
theme_minimal() +  # Use a minimal theme
theme(
plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Adjust title size and alignment
plot.subtitle = element_text(size = 14, hjust = 0.5),  # Adjust subtitle size and alignment
axis.text = element_text(size = 12),  # Adjust axis text size
axis.title = element_text(size = 14, face = "bold"),  # Adjust axis title size and weight
legend.title = element_blank(),  # Remove legend title
legend.text = element_text(size = 12),  # Adjust legend text size
plot.caption = element_blank()  # Remove caption
)
# Load ggplot2 package
library(ggplot2)
# Plot with enhanced aesthetics
ggplot(SESWithImputed, aes(x = SDQ_Hyperactivity, y = Entropy)) +
geom_point(color = "#0ff0fc", size = 3, alpha = 0.7) +  # Change color and transparency of points
geom_smooth(method = "glm", formula = y ~ x, color = "#ff8e4d", linetype = "dashed", size = 1.5, se = TRUE, alpha = 0.5, fill = "#b4fbff") +  # Change line style, color, and transparency of the trend line
labs(
x = "SDQ_Hyperactivity",
y = "Entropy",
title = "Scatter Plot of Entropy by SDQ_Hyperactivity",
subtitle = "with Trend Lines and Confidence Intervals"
) +
theme_minimal() +  # Use a minimal theme
theme(
plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Adjust title size and alignment
plot.subtitle = element_text(size = 14, hjust = 0.5),  # Adjust subtitle size and alignment
axis.text = element_text(size = 12),  # Adjust axis text size
axis.title = element_text(size = 14, face = "bold"),  # Adjust axis title size and weight
legend.title = element_blank(),  # Remove legend title
legend.text = element_text(size = 12),  # Adjust legend text size
plot.caption = element_blank()  # Remove caption
)
# Load ggplot2 package
library(ggplot2)
# Plot with enhanced aesthetics
ggplot(SESWithImputed, aes(x = MATRIX_Tscore, y = Entropy)) +
geom_point(color = "#0ff0fc", size = 3, alpha = 0.7) +  # Change color and transparency of points
geom_smooth(method = "glm", formula = y ~ x, color = "#ff8e4d", linetype = "dashed", size = 1.5, se = TRUE, alpha = 0.5, fill = "#b4fbff") +  # Change line style, color, and transparency of the trend line
labs(
x = "MATRIX_Tscore",
y = "Entropy",
title = "Scatter Plot of Entropy by MATRIX_Tscore",
subtitle = "with Trend Lines and Confidence Intervals"
) +
theme_minimal() +  # Use a minimal theme
theme(
plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Adjust title size and alignment
plot.subtitle = element_text(size = 14, hjust = 0.5),  # Adjust subtitle size and alignment
axis.text = element_text(size = 12),  # Adjust axis text size
axis.title = element_text(size = 14, face = "bold"),  # Adjust axis title size and weight
legend.title = element_blank(),  # Remove legend title
legend.text = element_text(size = 12),  # Adjust legend text size
plot.caption = element_blank()  # Remove caption
)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Specify the outcome variable
outcome_variable <- "Entropy"
# Specify potential predictors (including the outcome variable)
potential_predictors <- c("Age", "Gender", "SwitchingRate")
# Fit the GLM model
glm_model <- glm(get(outcome_variable) ~ Age + Gender + SwitchingRate, family = gaussian(link = "identity"), data = your_data)
# Extract coefficient estimates and confidence intervals
coef_df <- coef_plot_data(glm_model)
# Fit the GLM model
glm_model <- glm(Entropy ~ Age + Gender + SwitchingRate, family = gaussian(link = "identity"), data = your_data)
# Extract coefficient estimates and confidence intervals
coef_df <- data.frame(coef(summary(glm_model)))
View(glm_model)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "SwitchingRate"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "Entropy", "Subjective_SES")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + Entropy, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
ggplot(your_data, aes(x = SwitchingRate, y = Entropy)) +
geom_point(color = "#0ff0fc", size = 3, alpha = 0.7) +  # Change color and transparency of points
geom_smooth(method = "glm", formula = y ~ x, color = "#ff8e4d", linetype = "dashed", size = 1.5, se = TRUE, alpha = 0.5, fill = "#b4fbff") +  # Change line style, color, and transparency of the trend line
labs(
x = "SwitchingRate",
y = "Entropy",
title = "Scatter Plot of Entropy by SwitchingRate",
subtitle = "with Trend Lines and Confidence Intervals"
) +
theme_minimal() +  # Use a minimal theme
theme(
plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Adjust title size and alignment
plot.subtitle = element_text(size = 14, hjust = 0.5),  # Adjust subtitle size and alignment
axis.text = element_text(size = 12),  # Adjust axis text size
axis.title = element_text(size = 14, face = "bold"),  # Adjust axis title size and weight
legend.title = element_blank(),  # Remove legend title
legend.text = element_text(size = 12),  # Adjust legend text size
plot.caption = element_blank()  # Remove caption
)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "MATRIX_Tscore"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "FO1", "FO2", "FO3", "FO4", "FO5", "FO6", "FO7")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + FO1 + FO2 + FO3 + FO4 + FO5 + FO6 + FO7, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Equivalised_Income_Updated"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "FO1", "FO2", "FO3", "FO4", "FO5", "FO6", "FO7")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + FO1 + FO2 + FO3 + FO4 + FO5 + FO6 + FO7, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Subjective_SES"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "FO1", "FO2", "FO3", "FO4", "FO5", "FO6", "FO7")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + FO1 + FO2 + FO3 + FO4 + FO5 + FO6 + FO7, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Subjective_SES"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "SwitchingRate")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + MaxFO, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Subjective_SES"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "SwitchingRate")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + SwitchingRate, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Subjective_SES"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "MaxFO")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + MaxFO, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Subjective_SES"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "Entropy")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + Entropy, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Equivalised_Income_Updated"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "SwitchingRate")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + SwitchingRate, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Equivalised_Income_Updated"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "MaxFO")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + MaxFO, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
# Load your data
your_data <- read.csv("SESWithImputed.csv")
# Choose an outcome variable (e.g., SDQ score)
outcome_variable <- "Equivalised_Income_Updated"
# Choose potential predictors including nuisance regressors
potential_predictors <- c("Age", "Gender", "Entropy")
# Fit the GLM with Gaussian family and identity link function
glm_model <- glm(get(outcome_variable) ~ Age + Gender + Entropy, family = gaussian(link = "identity"), data = your_data)
# Summary of the model
summary(glm_model)
# Apply FDR correction
p_values <- summary(glm_model)$coefficients[, 4]
p_adjusted <- p.adjust(p_values, method = "fdr")
# Significant predictors after FDR correction
significant_indices <- which(p_adjusted < 0.05)
significant_predictors <- rownames(summary(glm_model)$coefficients)[significant_indices]
# Print significant predictors
print(significant_predictors)
