library(dplyr)
library(DALEX)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(ggraph)
library(igraph)

### 00 ZAPOZNANIE Z DANYMI #########

?titanic
dim(titanic)

View(titanic)

data("titanic_imputed")

View(titanic_imputed)

any(is.na(titanic_imputed))

str(titanic_imputed)

### Mamy factory -> w zaleznosci od tego jaki model uczenia maszynowego wybierzemy, 
### mozliwe ze beda one problemem

### 01 DEFINICJA PROBLEMU - KLASYFIKACJA #########

table(titanic_imputed$survived)/nrow(titanic_imputed)

## aby to byla klasyfikacja musimy wziac zmienic Y na factor
# titanic_imputed$survived <- as.factor(titanic_imputed$survived)

### 02 PREPROCESSING #############

# wcześniej uzupelnione zostały braki danych
# enconding zmiennych kategorycznych (zobaczymy, ze nie zawsze jest on niezbedny)

str(titanic_imputed)

is_categorical <- sapply(titanic_imputed, class)

titanic_ohe <- model.matrix(survived~., titanic_imputed) 

### Na poprzednich warsztatach budowaliśmy modele predykcyjne.
### Ogólnie naszym celem jest zbudowanie modeli o jak najlepszej mocy predykcyjnej 
### na nieznanych  danych pochodzących z tego samego mechanizmu generującego (ten sam rozkład danych)


### podzial na zbior testowy i treningowy
set.seed(123)
train_ratio <- 0.7
train_id <- sample(1:nrow(titanic_imputed), floor(train_ratio * nrow(titanic_imputed)))
test_id <- setdiff(1:nrow(titanic_imputed), train_id)


titanic_train <- titanic_imputed[train_id,]
titanic_test <- titanic_imputed[test_id,]

titanic_ohe_train <- titanic_ohe[train_id,]
titanic_ohe_test <- titanic_ohe[test_id,]



### 02 BUDOWANIE MODELU ####################


#### Regresja logistyczna ######
# model_glm <- glm(survived ~ class + gender + age + 
#       sibsp + parch + fare + embarked, data = titanic_train)


model_glm <- glm(as.factor(survived) ~ ., data = titanic_train)

### mozemy tez dzialac na orginalnej ramce danych ale z pakietem rms
### zaleta jest lepsza obsluga zmiennych kategorycznych (ale nie tylko!)

# install.packages('rms')
library(rms)
model_glm <- lrm(as.factor(survived) ~ ., data = titanic_train)

#### Drzewo decyzyjne #########

model_dt <- rpart(survived~., data = titanic_train)

#### Las losowy - random forest ############
library("randomForest")
set.seed(1313)
model_rf <- randomForest(as.factor(survived) ~ .,
                         data = titanic_train,
                         maxnodes = 2^4)

set.seed(1313)
model_rf_v3 <- randomForest(as.factor(survived) ~ class +gender + age, 
                            data = titanic_train,
                            maxnodes = 2^4)
model_rf_v3



### 03 INTERPRETOWALNOSC MODELI ####################

### Czego byśmy się chcieli dowiedzieć o działaniu modelu? 
#### Jakie zmienne są najistotniejsze i mają największy wpływ na zmianę predykcji
#### Jak poszczególne zmienne wpływają na predykcję modelu?
#### W przypadku określonych obserwacji dlaczego model podjął taką a nie inną decyzję?

### Interpretability is the degree to which a human can understand the cause of a decision. 
### Interpretability is the degree to which a human can consistently predict the model’s result 


### Jakie modele są uwazane za interpretowalne
### - regresja logistyczna/liniowa
### - kNN
### - drzewa decyzyjne



#### regresja logistyczna ############

model_glm

#### drzewo decyzyjne ##############
model_dt
#### wizualizacja
rpart.plot(model_dt)

names(model_dt)



#### Ale co jeśli mamy cały las? ############

class(getTree(model_rf, 1, labelVar=TRUE))


plot_tree_rf <- function(final_model, 
                      tree_num) {
  # browser()
  # get tree by index
  tree <- randomForest::getTree(final_model, 
                                k = tree_num, 
                                labelVar = TRUE) %>%
    tibble::rownames_to_column() %>%
    # make leaf split points to NA, so the 0s won't get plotted
    mutate(`split point` = ifelse(is.na(prediction), `split point`, NA))
  
  # prepare data frame for graph
  graph_frame <- data.frame(from = rep(tree$rowname, 2),
                            to = c(tree$`left daughter`, tree$`right daughter`))
  
  # convert to graph and delete the last node that we don't want to plot
  graph <- graph_from_data_frame(graph_frame) %>%
    delete_vertices("0")
  
  # set node labels
  V(graph)$node_label <- gsub("_", " ", as.character(tree$`split var`))
  V(graph)$leaf_label <- as.character(tree$prediction)
  V(graph)$split <- as.character(round(tree$`split point`, digits = 2))
  
  # plot
  plot <- ggraph(graph, 'dendrogram') + 
    theme_bw() +
    geom_edge_link() +
    geom_node_point() +
    geom_node_text(aes(label = node_label), na.rm = TRUE, repel = TRUE) +
    geom_node_label(aes(label = split), vjust = 2.5, na.rm = TRUE, fill = "white") +
    geom_node_label(aes(label = leaf_label, fill = leaf_label), na.rm = TRUE, 
                    repel = TRUE, colour = "white", fontface = "bold", show.legend = FALSE) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.background = element_blank(),
          plot.background = element_rect(fill = "white"),
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(size = 18))
  
  plot
}

plot_tree_rf(model_rf, tree_num = 3)

### Co zrobić dla bardziej skomplikowanych modeli?
### Jak to porównywać pomiędzy różnymi typami modeli?


### 04 Model agnostic methods ############

#### przydatny pakiet DALEX - ujednolicenie interfejsu modeli
### wazne w przypadku klasyfikacji: y - powinno być zmienna numeryczna 0/1 lub zmienna logiczna


exp_glm_train <- DALEX::explain(model_glm, data = titanic_train[,-8],
                         y = titanic_train$survived,
                         label = 'LogisticReg')

exp_dt_train <- DALEX::explain(model_dt, data = titanic_train[,-8],
                         y = titanic_train$survived,
                        label = 'DecisionTree')

exp_rf_train <- DALEX::explain(model_rf, data = titanic_train[,-8],
                        y = titanic_train$survived,
                        label = 'RandomForestAll')

exp_rfv3_train <- DALEX::explain(model_rf_v3, data = titanic_train[,-8],
                        y = titanic_train$survived,
                        label = 'RandomForestV3')

## modyfikacja funkcji prediction_function, residual_function

### 04a metody lokalne ############

henry <- data.frame(
  class = factor("1st", levels = c("1st", "2nd", "3rd", 
                                   "deck crew", "engineering crew", 
                                   "restaurant staff", "victualling crew")),
  gender = factor("male", levels = c("female", "male")),
  age = 47, sibsp = 0, parch = 0, fare = 25,
  embarked = factor("Cherbourg", levels = c("Belfast",
                                            "Cherbourg","Queenstown","Southampton")))
henry

johnny_d <- data.frame(
  class = factor("1st", levels = c("1st", "2nd", "3rd", 
                                   "deck crew", "engineering crew", 
                                   "restaurant staff", "victualling crew")),
  gender = factor("male", levels = c("female", "male")),
  age = 8, sibsp = 0, parch = 0, fare = 72,
  embarked = factor("Southampton", levels = c("Belfast",
                                              "Cherbourg","Queenstown","Southampton")))

johnny_d

### Predykcje dla tej obserwacji

pred_henry_glm <- predict(model_glm, newdata = henry, type = 'fitted')
### jesli stosuje sie glm to: type = 'link' 
pred_henry_dt <- predict(model_dt, newdata = henry)
pred_henry_rf <- predict(model_rf, newdata = henry, type = 'prob')


#### Jaki jest wklad poszczegolnych zmiennych w predykcje?

#### Breakdown values #########
magick::image_read('https://ema.drwhy.ai/figure/break_down_distr.png')

bd_rf <- predict_parts(explainer = exp_rf_train,
                       new_observation = henry,
                       type = "break_down")
bd_rf 

plot()


### zmiana kolejnosci warunkowania

bd_rf_order <- predict_parts(explainer = exp_rf_train,
                             new_observation = henry, 
                             type = "break_down",
                             order = c("class", "age", "gender", "fare", 
                                       "parch", "sibsp", "embarked"))
plot(bd_rf_order) 

### zostawiamy informacje o rozkladzie
bd_rf_order <- predict_parts(explainer = exp_rf_train,
                             new_observation = henry, 
                             type = "break_down",
                             order = c("class", "age", "gender", "fare", 
                                       "parch", "sibsp", "embarked"))
plot(bd_rf_order)

#### SHAP values ###########
magick::image_read('https://ema.drwhy.ai/figure/shap_10_replicates.png')

sh_rf <- predict_parts(explainer = exp_rf_train,
                       new_observation = henry,
                       type = "shap")
sh_rf <- predict_parts(explainer = exp_rf_train,
                       new_observation = henry,
                       type = "shap",
                       B = 25)

sh_rf 

plot(sh_rf)
plot(sh_rf, max_features = 3)

#### Interakcje? ####


bd_int_rf <- predict_parts(explainer = exp_rf_train,
                       new_observation = johnny_d,
                       type = "break_down_interactions",
                       order = c("class", "age", "gender", "fare", 
                                 "parch", "sibsp", "embarked"))
plot(bd_int_rf)

bd_rf_order <- predict_parts(explainer = exp_rf_train,
                           new_observation = johnny_d,
                           type = "break_down",
                           order = c("class", "age", "gender", "fare", 
                                     "parch", "sibsp", "embarked"))
plot(bd_rf_order)

#### Surrogate model: LIME #######
### przyblizamy skomplikowany model lokalnie modelem liniowym
# magick::image_read('https://ema.drwhy.ai/figure/lime_introduction.png')

library("DALEXtra")
library("lime")

model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_henry <- DALEXtra::predict_surrogate(explainer = exp_rf_train, 
                                 new_observation = henry, 
                                 # n_features = 3, 
                                 # n_permutations = 1000,
                                 type = "lime")
plot(lime_henry)


### Jak zmieni sie predykcja dla tej obserwacji jesli zmieni sie wartosc zmiennej?

#### Individual Conditional Expectations (Ceteris Paribus profile)

cp_rf <- predict_profile(explainer = exp_rf_train, 
                                 new_observation = henry)

cp_rf

plot(cp_rf)

plot(cp_rf, variables = c('age', 'fare'))

### zmienne kategoryczne

plot(cp_rf, variables = c("class", "embarked"), 
     variable_type = "categorical", categorical_type = "bars") +
  ggtitle("Ceteris-paribus profile", "") 


### istotne jest sprawdzenie rozkladu badanej zmiennej

ggplot(titanic_train)+
  geom_histogram(aes(x = age))

ggplot(titanic_train)+
  geom_histogram(aes(x = fare))

### variable split

plot()

### porownanie roznych modeli

cp_rfv3 <- predict_profile(explainer = exp_rfv3_train, 
                           new_observation = henry)
plot(cp_rfv3, cp_rf)


### 04b metody globalne ############
# podsumowanie działania modelu z perspektywy całych danych i poszczególnych zmiennych

### Jak dobry jest model? #####
model_performance(exp_rf_train)
model_performance(exp_rfv3_train)


###  Model specific methods #######

model_dt$variable.importance

model_rf$importance

#### Variable importance ######


vi_glm <- model_parts(exp_glm_train)
plot(vi_glm)

vi_dt <- model_parts(exp_dt_train)
vi_rf <- model_parts(exp_rf_train)

plot(vi_glm, vi_dt, vi_rf)
plot(vi_rf)


### Jak liczona jest funkcja straty?
vi_rf_ratio <- model_parts(exp_rf_train, type = 'ratio')
plot(vi_rf_ratio) 

### Jezeli za dlugo sie liczy mozna ograniczyc liste zmiennych albo ilosc permutacji


### SHAP Feature importance



#### Partial dependence plot ##########

pdp_glm <- model_profile(exp_glm_train, type = 'partial')
plot(pdp_glm)


pdp_dt <- model_profile(exp_dt_train, type = 'partial')
pdp_rf <- model_profile(exp_rf_train, type = 'partial')

plot(pdp_glm, pdp_dt, pdp_rf)

#### ALE plot ##########

ale_glm <- model_profile(exp_glm_train, type = 'accumulated')
plot(ale_glm)


ale_dt <- model_profile(exp_dt_train, type = 'accumulated')
ale_rf <- model_profile(exp_rf_train, type = 'accumulated')

plot(ale_glm, ale_dt, ale_rf)


### modelStudio

library(modelStudio)
modelStudio()

#### Wiecej informacji, przykladow i teorii: https://ema.drwhy.ai/
