###################################################
########       Warsztaty Data Science      ########
####     Eksploracja i Feature Engineering     ####
###################################################

# Pakiety
library(DALEX)
library(dplyr)
library(tidyr)
library(forcats)
library(ggplot2)

# 0) Wstęp

# Eksploracyjna analiza danych jest cyklem iteracyjnym:
# 1) Generuj pytania dotyczące Twoich danych.
# 2) Znajdź odpowiedź poprzez wizualizację, przekształcenie lub modelowanie danych.
# 3) Zdobytą wiedzę wykorzystaj do dopracowania pytań i/lub wygenerowania nowych.

# EDA nie jest formalnym procesem o ścisłym zestawie reguł, ale jest ważną częścią
# ponieważ pozwala na zbadanie jakości danych.

# Zrozumienie zbioru danych:
#   - wyodrębnienie ważnych zmiennych i pozostawienie bezużytecznych zmiennych,
#   - identyfikacja wartości odstających, brakujących wartości lub błędów ludzkich,
#   - zrozumienie zależności lub ich braku pomiędzy zmiennymi.

?titanic

# 1) Kilka informacji o zbiorze danych

## a) wymiar ramki danych
dim()

## b) kilka pierwszych wierszy
head()

## c) kilka ostatnich wierszy 
tail()

## d) jakie mamy kolumny
columns()
names()

## e) liczność, średnia, odchylenie, min, max dla każdej zmiennej, liczba braków danych
summary()

## f) ile mamy unikalnych wartośc
apply(titanic, 2, funtion(x){length(unique(x))})


# 2) Eksporacja danych

## A. Zmienne jakościowe
### binarne (dwie wartości) np. płeć
### nominalne (jakościowe nieuporządkowane) np. marka smachodu
### uporządkowane (jakościowe uporządkowane) np. wykształcenie: podstawowe/średnie/wyższe


## B. Zmienne ilościowe
### zliczenia (liczba wystąpień pewnego zjawiska, opisywana liczbą całkowitą) np. liczba lat nauki, liczba wypadków
### ilorazowe (zmienne mierzone w skali, w której można dzielić wartości - ilorazy mają sens) np. długość w metrach 
### przedziałowe (mierzone w skali, w której można odejmować wartości) np. daty lub stopnie temperatury 

## Jak sprawdzić rodzaj zmiennej?
str(titanic)


## Czy żyje? - kluczowa zmienna, bo dla niej będziemy chcieli zrobić predykcję.

table(titanic$survived)

ggplot(titanic, aes(x = survived)) +
  geom_bar() + 
  labs(x = "Czy żyje?", y = "Częstość")

## procent tych, którzy przeżyli
sum(titanic$survived == "yes")/nrow(titanic)

## Ad A.
# tabela
# wykres słupkowy

# zmienna gender
table(titanic$gender)

ggplot(titanic, aes(x = gender)) +
  geom_bar() +
  labs(x = "Płeć", y = "Częstość", title = "Rozkład zmiennej płeć") + 
  scale_y_continuous(expand = c(0,0))

# zmienna gender względem survived

ggplot(titanic, aes(x = gender, color = survived)) +
  geom_bar() +
  labs(x = "Płeć", y = "Częstość", title = "Rozkład zmiennej płeć") + 
  scale_y_continuous(expand = c(0,0))

ggplot(titanic, aes(x = gender, fill = survived)) +
  geom_bar() +
  labs(x = "Płeć", y = "Częstość", title = "Rozkład zmiennej płeć", fill = "Czy żyje?") 


ggplot(titanic, aes(x = gender, fill = survived)) +
  geom_bar(position = "dodge") +
  labs(x = "Płeć", y = "Częstość", title = "Rozkład zmiennej płeć", fill = "Czy żyje?") 

#### Zadanie 1 ####
#Sprawdzić rozkład zmiennej class.

table(titanic$class)

ggplot(titanic, aes(x = class)) +
  geom_bar()

titanic <-  titanic %>% 
  mutate(class_new = ifelse(titanic$class == "1st" | titanic$class == "2nd" | titanic$class == "3rd", titanic$class, "other"))

titanic %>% 
  group_by(class_new) %>% 
  summarise(n= n()) %>% 
  ggplot(aes(x = class_new, y = n)) +
  geom_col() 



## Ad B.
# histogram
# density plot
# boxplot

ggplot(titanic, aes(x = age)) +
  geom_histogram()

ggplot(titanic, aes(x = age)) +
  geom_histogram(binwidth = 5)

ggplot(titanic, aes(x = age)) +
  geom_boxplot()

ggplot(titanic, aes(x = age, fill = gender)) +
  geom_density(alpha = 0.2)


ggplot(titanic, aes(x = age, fill = class_new)) +
  geom_density(alpha = 0.2)


## Zależność dwóch zmiennych - jak badać?
# a) Jakościowa i ilościowa:
#     - freqpoly()
#     - geom_boxplot()
#     - geom_violin()




# b) Dwie zmienne jakościowe:
#     - geom_count()
#     - geom_tile()



# c) Dwie zmienne ilościowe:
#     - geom_point()
#     - geom_bin2d()
#     - geom_hex()

### Automaczyna EDA

## *) dlookr
# https://github.com/choonghyunryu/dlookr
install.packages("dlookr")
library(dlookr)
library(dplyr)
summary(mtcars)

describe(mtcars)

mtcars %>% 
  describe() %>% 
  select(variable, mean, IQR, p50)

normality(mtcars)
plot_normality(mtcars, mpg)

plot_normality(mtcars, cyl)

plot_correlate(mtcars)


## *) dataReporter
# https://github.com/ekstroem/dataMaid
# https://github.com/ekstroem/dataReporter
install.packages("dataReporter")
library("dataReporter")

makeDataReport(mtcars)
data("mtcars")



#### Feature Engineering (Inżynieria cech)

## Braki danych
# (age) -- średnia
# sibsp, parch -- najczęstsza wartość

## Outliers

## Normalizacja 
(x - x_min)/(x_max - x_min)

## Standaryzacja 

zVar <- (myVar - mean(myVar)) / sd(myVar)

## One hote encoding
library(caret)
dmy <- dummyVars(" ~ .", data = titanic)
trsf <- data.frame(predict(dmy, newdata = titanic))

## Podział zbioru danych na treningowy i testowy 

## 75% zbioru danych
smp_size <- floor(0.75 * nrow(mtcars))

set.seed(123)
train_ind <- sample(seq_len(nrow(mtcars)), size = smp_size)

train <- titanic[train_ind, ]
test <- titanic[-train_ind, ]
