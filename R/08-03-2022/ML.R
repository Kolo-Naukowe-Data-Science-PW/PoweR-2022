###################################################
########       Warsztaty Data Science      ########
####         Podstawy Machine Learning         ####
###################################################

# install.packages("dplyr")
# install.packages("tidyr")
# install.packages("forcats")
# install.packages("ggplot2")
# install.packages("randomForest")
# install.packages("DALEX")
# install.packages("party") # ctrees
# install.packages("ROCR")

# Pakiety
library(dplyr)
library(tidyr)
library(forcats)
library(ggplot2)
library(caret)



### Ładowanie danych ###
# W danych kagglowych zbiór ten jest podzielony na 3 części, które musimy załadować

raw_train <- read.csv('train.csv')
raw_test  <- read.csv('test.csv')
y_test  <- read.csv('gender_submission.csv')

raw_test$Survived = y_test$Survived
titanic  <- bind_rows(raw_train, raw_test)


#####################################
###   Krótka eksploracja danych   ###
#####################################

### Jakie dane posiadamy?
### Jakie mamy zadanie? klasyfikacji czy eksploracji?


head(titanic, 5)


str(titanic)


## Odrzucenie niestotnych zmiennych

titanic <- subset(titanic, select = -c(PassengerId, Ticket))

# sprawdźmy kolumny
colnames(titanic)




### Detekcja braków

# Chcemy znaleźć wszystkie przypadki, gdzie posiadamy braki informacji
# lub nieprawidłowe wartosci w kolumnach

colSums(is.na(titanic)) # liczymy puste komórki w ramce

colSums(titanic=='') # liczymy komórki z pustym stringiem jako wartość 


# Wnioski:
# Wykryliśmy braki w kolumnach Age, Fare, Cabin i Embarked





### Spojrzenie na zmienną celu


# Ile osób przeżyło, a ile nie?
table(titanic$Survived) # 0 oznacza, że osoba nie przeżyła

# Podobno statek opuszczają najpierw kobiety i dzieci
# czy to powiedzenie ma odzwierdziedlenie też w danych?

table(titanic$Survived, titanic$Sex)


ggplot(titanic, aes(x = Age, fill = factor(Survived))) + 
  geom_histogram(position = "dodge", binwidth = 3)





###############################
###   Feature Engineering   ###
###############################

### Musimy zająć się brakami w danych
### Powinniśmy zakodować zmienne tak by móc wyszkolić na nich model



### Wcześniej znaleźliśmy braki, co z nimi zrobić?

### Podejście 1 - usuwanie braków (kolumna Cabin)

clean_titanic <- na.omit(titanic)
clean_titanic <- clean_titanic[clean_titanic$Cabin != '',]

nrow(clean_titanic)

# Z reguły usuwa nam znaczącą ilość wierszy, czasami lepiej usunąć tylko
# problematyczne kolumny, takie jak np kolumna Cabin

proc <- nrow(titanic[titanic$Cabin=='',])/nrow(titanic)
cat(floor(proc*100),"% danych kolumny Cabin to pusty string")

titanic <- subset(titanic, select = -c(Cabin))


### Podejście 2 - imputacja danych

colSums(is.na(titanic))

## Uzupełnianie zmiennych liczbowych (kolumny Age i Fare)
### Uzupełnienie statystyką 

# średnia
mean(titanic$Age, na.rm = TRUE)

# mediana
age_median = median(titanic$Age, na.rm = TRUE)
age_median

## nadpisywanie komórek kolumny Age jej medianą
titanic$Age[is.na(titanic$Age)] <- age_median


## to samo dla kolumny Fare
titanic$Fare[is.na(titanic$Fare)] <- median(titanic$Fare, na.rm = TRUE)


## Uzupełnianie zmiennych kategorycznych (kolumna Embarked)

titanic[titanic$Embarked =='',]

table(titanic$Embarked, titanic$Sex)

titanic$Embarked[titanic$Embarked ==''] <- 'S'



## Sprawdzamy czy na pewno udało nam się usunąć wszystkie braki danych
colSums(is.na(titanic)) 
colSums(titanic=='') 




### Przekształcenie/kodowanie zmiennych
# przykłady różnych kodowań
# https://www.r-bloggers.com/2020/02/a-guide-to-encoding-categorical-features-using-r/



### Odzyskanie informacji o tytule 

titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')


titanic$Title[titanic$Title == 'Mlle']        <- 'Miss' 
titanic$Title[titanic$Title == 'Ms']          <- 'Miss'
titanic$Title[titanic$Title == 'Mme']         <- 'Mrs' 
titanic$Title[titanic$Title %in% rare_title]  <- 'Rare_Title'


table(titanic$Sex, titanic$Title)

## Odrzucenie niepotrzebnych zmiennych
titanic <- subset(titanic, select = -c(Name))




### One hot encoding zmiennych kategorycznych

## konwersja na factory
to_factor <- c("Survived","Pclass","Sex","Embarked","Title")
titanic[to_factor] <- lapply(titanic[to_factor], factor)

survived <- titanic[Survived, Sex] # tych kolumn nie chcemy kodować  

to_save <- c("Survived","Sex")

saved <- titanic[to_save]

x <- dummyVars("  ~ . - Survived - Sex", data=titanic)
titanic <- as.data.frame(predict(x, newdata = titanic))

titanic[to_save] <- saved
titanic$Sex  <- factor(as.integer(titanic$Sex == "female")) ## encoding female as 1





##### Dzielenie zbiorów na uczący (treningowy) i testowy

## normalnie podzielili byśmy zbiór tak jak poniżej, ale titanic ma już określony podział
# train_size <- floor(0.7 * nrow(titanic))
# set.seed(123)
# train_ind <- sample(seq_len(nrow(titanic)), size = train_size)

n <- nrow(raw_train)
X_train <- titanic[1:n,]
X_test <- titanic[n:nrow(titanic),]






###############################
###    Trenowanie modelu    ###
###############################



### RandomForest

## Idea lasu losowego -> losowo tworzone drzewa decyzyjne

## Przykład drzewa decyzyjnego
library(party)
output.tree <- ctree(Survived~., data=X_train)

plot(output.tree)


## Lasy losowe
library(randomForest)

?randomForest


## wykorzystajmy las losowy by otrzymać predykcje!
rf <- randomForest(formula = Survived~., data=X_train)

rf$type


## Look at variable importance:
round(importance(rf), 2)
varImpPlot(rf)


### 'Strojenie' modelu (Hyperparameter tiuning) 

## funkcje trainControl i train z pakietu carpet

control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='grid')

?randomForest
tunegrid <- expand.grid(mtry = seq(1,17,3)) 



rf_gridsearch <- train(Survived ~ ., 
                       data = X_train,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid)

print(rf_gridsearch)

rf_gridsearch$finalModel

rf$mtry






### Ewaluacja modelu ###

# Błąd modelu w zależności od ilości drzew
plot(rf, ylim=c(0,0.36))
legend('topright', colnames(rf$err.rate), col=1:3, fill=1:3)


# przewidujemy los osób z zbioru testowego
ypred_class <-  predict(rf, X_test, type = "class")


# dla porównania zróbmy głupią predykcje, że wszyscy umrą
bad_pred <- factor(rep(0, length(ypred_class)))


confusionMatrix(ypred_class, X_test$Survived) # wynik lasu losowego
confusionMatrix(bad_pred, X_test$Survived) # wynik naszego depresyjnego założenia



### Krzywa ROC i AUC ROC
library(ROCR)

# tym razem patrzymy na prawdopodobieństwa zdarzeń
ypred <-  predict(rf, X_test, type = "prob")

head(ypred)

ypred <- ypred[,2]


## tworzenie krzywej ROC

pred_ROCR <- prediction(ypred, X_test$Survived)
pref <- performance(pred_ROCR,"tpr","fpr")


plot(pref)
abline(a=0, b= 1)
title("Krzywa ROC dla modelu RandomForest")

## Pole pod krzywą ROC (Area Under the Curve ROC)

auc.hidden <- performance(pred_ROCR,"auc");
as.numeric(auc.hidden@y.values)



