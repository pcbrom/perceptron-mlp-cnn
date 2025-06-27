# Imports + setup -------------------------------------------------------------

# wd
setwd("/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/perceptron-mlp-cnn")

# libraries
library(tidyverse)
library(caret)

# data
db_full <- read_csv("data/df_clean.csv") |> mutate(Outcome = as.factor(Outcome))
db_train <- read_csv("data/train.csv") |> mutate(Outcome = as.factor(Outcome))
db_test <- read_csv("data/test.csv") |> mutate(Outcome = as.factor(Outcome))
db_validation <- read_csv("data/validation.csv") |> mutate(Outcome = as.factor(Outcome))


# Avaliação do balanceamento de classes ---------------------------------------

get_prop <- function(x) x |> table() |> prop.table()
get_prop(db_full$Outcome)
get_prop(db_train$Outcome)
get_prop(db_test$Outcome)
get_prop(db_validation$Outcome)

# usar em ambos os modelos:
# * Glucose - Glicemia de jejum (ou teste capilar)
# * BMI - IMC
# * Age - Idade
# * Pregnancies - Número de gestações

get_cor <- function(x) {
  x |> 
    select("Glucose", "BMI", "Age", "Pregnancies") |> 
    cor() |> 
    as.data.frame() |>
    rownames_to_column("Variable") |> 
    pivot_longer(-Variable, names_to = "Variable2", values_to = "Correlation")
}
joint <- function(x, y) {
  x |> 
    left_join(y, by = c("Variable", "Variable2"))
}
joint(get_cor(db_full), get_cor(db_train)) |> 
  joint(get_cor(db_test)) |> 
  joint(get_cor(db_validation)) |> 
  rename(
    Full = Correlation.x, 
    Train = Correlation.y, 
    Test = Correlation.x.x, 
    Validation = Correlation.y.y
  )


# Regressão Logística - Abordagem Estatística - Modelo Explicativo ------------







# Regressão Logística - Abordagem Estatística - Modelo Preditivo --------------






