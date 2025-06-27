# Imports + setup -------------------------------------------------------------

# wd
setwd("/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/perceptron-mlp-cnn")

# libraries
library(tidyverse)
library(caret)
library(ROSE)
library(ResourceSelection)
library(rms)
library(stats)

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


# Regressão Logística - Abordagem Estatística ---------------------------------

# Função para calcular R2 de McFadden
r2 <- function(model) 1 - (model$deviance / model$null.deviance)

# Modelo explicativo - Regressão Logística
# Pregnancies não significativa
# Efeito de interação não significativo

model_logistic_A <- glm(
  Outcome ~ Glucose + BMI + Age, 
  data = db_full, 
  family = binomial(link = "logit")
)
anova(model_logistic_A, test = "Chisq")
summary(model_logistic_A)

# R2 de McFadden
r2(model_logistic_A)

# Escolha de modelo sem transformação de dados
# - Treinamento de equipes de enfermagem costuma preferir coeficientes diretos em
#   mg dL⁻¹ e kg m⁻².
# - Políticas de saúde pública utilizam pontos de corte absolutos 
#   (≥ 126 mg dL⁻¹ para glicemia em jejum, IMC ≥ 30 kg m⁻²).
# - A recalibração intercepto + inclinação é necessária em ambos; não altera a
#   ordem de preferência.


# Recalibração
# | Aspecto       | Justificativa                                                                                                                                                                                                      | Consequência prática                                                                                                               |
# | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
# | Calibração    | O teste de Hosmer–Lemeshow indicou discordância sistemática entre probabilidades previstas e observadas para os dois modelos lineares. Isso revela viés global (intercepto) e/ou distorção de escala (inclinação). | Ajustar apenas intercepto (γ₀) e inclinação (γ₁) sobre o logito original corrige o desvio médio sem reconstruir todo o modelo. |
# | Discriminação | O procedimento não modifica o ranking de risco (AUC permanece idêntica).                                                                                                                                           | O poder de separação dos modelos não muda; portanto o ordenamento “modelo mais explicável” (escala original) continua válido.      |
# | Parcimônia    | Recalibração pós-estime consome 2 g.l., muito menos que acrescentar termos spline em um modelo já preferido.                                                                                                       | Mantém a simplicidade interpretativa e computacional, preservando a escolha do Modelo A.                                           |
  
# Obter logits e probabilidades
logit_orig <- predict(model_logistic_A, newdata = db_full, type = "link")
prob_orig  <- plogis(logit_orig)

# Calibração do modelo (intercept + slope)
cal_model <- glm(Outcome ~ logit_orig, family = binomial, data = db_full)

# Obter coeficientes de calibração
gamma0 <- coef(cal_model)[1]  # intercept
gamma1 <- coef(cal_model)[2]  # slope

# Gerar probabilidades calibradas
prob_recal <- plogis(gamma0 + gamma1 * logit_orig)

# Configuração do datadist (necessário para rms)
dd <- datadist(db_full)
options(datadist = "dd")

# Ajuste do modelo com splines cúbicas restritas
fit_rcs <- lrm(
  Outcome ~ rcs(Glucose, 4) + rcs(BMI, 4) + Age,
  data = db_full,
  x = TRUE,   # armazena matriz de preditores
  y = TRUE    # armazena vetor resposta
)

# Calibração interna via bootstrap (B = 1000)
cal <- calibrate(fit_rcs, method = "boot", B = 1000)

# Gráfico de calibração
png("figures/calibration_plot.png", width = 800, height = 800, res = 150)
plot(
  cal,
  xlab = "Probabilidade prevista",
  ylab = "Probabilidade observada",
  main = "Curva de calibração do modelo preditivo",
)
dev.off()

# Avaliação do modelo
anova(fit_rcs)
summary(fit_rcs)

# Conclusões
# Escolha do melhor modelo para explicação clínica
# - Parcimônia: O modelo linear utiliza menos parâmetros, apresenta AIC apenas 
#   5,5 unidades maior e pseud-R² quase idêntico.
# - Interpretabilidade: Odds ratios por unidade física são comunicáveis sem
#   gráficos adicionais, facilitando protocolos de enfermagem e decisões
#   baseadas em pontos de corte.
# - Calibração: Após recalibração intercepto + inclinação, o modelo linear
#   alcançou o mesmo MAE que o RCS (0,014) e mostrou curva próxima da ideal.
# - Portanto, recomenda-se manter o Modelo Linear (Glucose, BMI, Age em escala
#   original) como ferramenta explicativa principal.
# - O ajuste RCS pode ser arquivado como análise de sensibilidade, evidenciando
#   que a suposição de linearidade é adequada e que não há ganho clínico
#   relevante ao adotar maior complexidade.



# Avaliação da qualidade preditiva do modelo ----------------------------------

# Qualidade preditiva do modelo A

# Função para pegar métricas
get_metrics <- function(model, data, tipo) {
  pred_logistic <- predict(model, newdata = data, type = "response")
  pred_logistic_class <- ifelse(pred_logistic > 0.5, 1, 0) |> as.factor()
  print(confusionMatrix(pred_logistic_class, data$Outcome))
  roc.curve(data$Outcome, pred_logistic, main = paste0("ROC Curve: ", tipo))
}

get_metrics(model_logistic_A, db_train, "Treino")
get_metrics(model_logistic_A, db_test, "Teste")
get_metrics(model_logistic_A, db_validation, "Validação")


# Qualidade preditiva do modelo recalibrado

# logistic linear
prob_lin  <- predict(model_logistic_A, newdata = db_full, type = "response")

# logistic linear recalibrado
logit_lin <- predict(model_logistic_A, newdata = db_full, type = "link")
prob_cal  <- plogis(gamma0 + gamma1 * logit_lin)

# modelo com splines
get_metrics_rcs <- function(model, data, tipo) {
  pred_rcs <- predict(model, newdata = data, type = "fitted")
  pred_rcs_class <- ifelse(pred_rcs > 0.5, 1, 0) |> as.factor()
  print(confusionMatrix(pred_rcs_class, data$Outcome))
  roc.curve(data$Outcome, pred_rcs, main = paste0("ROC Curve: ", tipo))
}
get_metrics_rcs(fit_rcs, db_train, "Treino")
get_metrics_rcs(fit_rcs, db_test, "Teste")
get_metrics_rcs(fit_rcs, db_validation, "Validação")

# Conclusões:
# | Métrica           | Linear Treino | Linear Teste | Linear Validação | RCS Treino | RCS Teste | RCS Validação |
# | ----------------- | ------------- | ------------ | ---------------- | ---------- | --------- | ----------------- |
# | Acurácia          | 0,788         | 0,797        | 0,831            | 0,777      | 0,780     | 0,831             |
# | Kappa             | 0,495         | 0,507        | 0,622            | 0,470      | 0,474     | 0,612             |
# | Sensibilidade     | 0,896         | 0,900        | 0,872            | 0,885      | 0,875     | 0,897             |
# | Especificidade    | 0,571         | 0,579        | 0,750            | 0,560      | 0,579     | 0,700             |
# | Balanced Accuracy | 0,734         | 0,740        | 0,811            | 0,723      | 0,727     | 0,799             |
# | AUC               | 0,832         | 0,887        | 0,863            | 0,839      | 0,903     | 0,869             |
# | McNemar p-valor   | 0,0126        | 0,386        | 1,000            | 0,021      | 0,579     | 0,752             |
  
# Síntese objetiva
# - O desempenho discriminatório é elevado em ambos os modelos (AUC ≥ 0,83), 
#   com variações ≤ 0,02 entre metodologias nas três partições.
# - O modelo linear mantém especificidade mais alta na validação
#   (0,75 vs. 0,70) e acurácia idêntica ao RCS (0,831).
# - O modelo RCS oferece pequena vantagem em AUC na amostra de teste 
#   (0,903 vs. 0,887), porém não sustenta melhoria consistente em treino
#   ou validação.
# - McNemar indica leve assimetria de erros somente no treino; fora dele, ambos
# concentram p-valores > 0,05.

# Dado que o ganho preditivo do RCS é modesto e os termos não-lineares não
# alcançaram significância, o modelo linear continua preferível para finalidade
# explicativa e implementação clínica.

# Do ponto de vista estritamente preditivo, a diferença de desempenho não
# justifica o aumento de complexidade:
# - Modelo Linear oferece maior balanced accuracy e igual acurácia na validação,
#   com menor risco de superajuste.
# - Modelo RCS eleva AUC marginalmente, porém cai em especificidade, fator
#   crítico quando se quer evitar sobre-encaminhamentos.

# Portanto, para implementação em triagem populacional, recomenda-se manter
# o modelo linear.