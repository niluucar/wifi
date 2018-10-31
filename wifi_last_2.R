install.packages("dplyr")
install.packages("ggplot2")
install.packages("ggmap")
install.packages("parallel")
install.packages("doParallel")
install.packages("caret")
install.packages("dplyr")
install.packages("randomForest")

library("randomForest")
library("dplyr")
library("caret")
library("doParallel")
library("parallel")
library("dplyr")
library("ggplot2")
library("ggmap")
library("lubridate")

setwd("C:/Users/Lenovo/Desktop/Ubiqum_data/Wi-fi")
library(readr)
trainingData <- read_csv("C:/Users/Lenovo/Desktop/Ubiqum_data/Wi-fi/trainingData.csv")
ValidationData <- read_csv("C:/Users/Lenovo/Desktop/Ubiqum_data/Wi-fi/validationData.csv")

td<-trainingData
vd<-ValidationData
td$TIMESTAMP<- as_datetime(td$TIMESTAMP)
vd$TIMESTAMP<- as_datetime(vd$TIMESTAMP)
str(td)
summary(td[,1:52])

####Changing the data types####

td$FLOOR<-as.factor(td$FLOOR)
td$BUILDINGID<-as.factor(td$BUILDINGID)
td$SPACEID<-as.factor(td$USERID) # 18 different points.  Internal ID number to identify the Space (office, corridor, classroom) where the capture was taken, 
td$RELATIVEPOSITION<-as.factor(td$RELATIVEPOSITION) # in vs out
td$USERID<-as.factor(td$USERID)
td$PHONEID<-as.factor(td$PHONEID)
#The same for Validation Data: 
vd$FLOOR<-as.factor(vd$FLOOR)
vd$BUILDINGID<-as.factor(vd$BUILDINGID)
vd$SPACEID<-as.factor(vd$USERID)
vd$RELATIVEPOSITION<-as.factor(vd$RELATIVEPOSITION)
vd$USERID<-as.factor(vd$USERID)
vd$PHONEID<-as.factor(vd$PHONEID)


####CHANGING THE 100 TO -105 AND REMOVING THE REPEATED ROWS ####

td<-distinct(td)
td[td==100]<--105

vd<-distinct(vd)
vd[vd==100]<--105


#### counting the values which are not equal to 100####
td$count <- apply(td[,1:520], 1, function(x) length(which(x!=-105)))
td$max <- apply(td[,1:520], 1, max)
td$max_2<-apply(td[,1:520],1,function(x) names(td[,1:520])[which(x==max(x))])
td$max3<-apply(td[1:520],1,function(x) names(which.max(x)))
#vd #
vd$count <- apply(vd[,1:520], 1, function(x) length(which(x!=-105)))

vd$max <- apply(vd[,1:520], 1, max)
vd$max_2<-apply(vd[,1:520],1,function(x) names(vd[,1:520])[which(x==max(x))])
vd$max3<-apply(vd[1:520],1,function(x) names(which.max(x)))
#removing the rows ###

td<-subset(td,td$max!=0 & td$count!=0)

td<-subset(td,td$max<=-30 & td$max>=-80 )

#training dataset remove 1 value columns
waps_td<-td[,c(1:520)]
useless_waps<-apply(waps_td, 2, function(x) length(unique(x))==1)
td_new<-td[,-c(which(useless_waps==TRUE))]

waps_vd<-vd[,c(1:520)]
useless_waps_vd<-apply(waps_vd, 2, function(x) length(unique(x))==1)
vd_new<-vd[,-c(which(useless_waps_vd==TRUE))]
#identifying WAPS training
Waps_td_names <- grep("WAP", names(td_new), value = TRUE)

#identifying WAPS VALIDATION

Waps__vd_names <- grep("WAP", names(vd_new), value = TRUE)

Waps_tdvd <- intersect(Waps_td_names, Waps__vd_names)

x <- names(td_new[Waps_td_names]) %in% Waps_tdvd == FALSE
td_new_2 <- td_new[-which(x)]

#remove columns
y <- names(vd_new[Waps__vd_names]) %in% Waps_tdvd  == FALSE
vd_new_2 <- vd_new[-which(y)]

####BUILDING PREDICTION ####
detectCores()
clusterF1 <- makeCluster(detectCores()-1)
registerDoParallel(clusterF1)


td_build<- select(td_new_2,-c(TIMESTAMP,USERID,PHONEID,max_2,max,count,SPACEID,FLOOR,
                        RELATIVEPOSITION,LATITUDE,LONGITUDE,max3))

## Model
set.seed(124)

fitControl <- trainControl(method = "repeatedcv", number=3,repeats = 3, 
                           verboseIter = TRUE, allowParallel = TRUE)

## knn
knnFit <- train(BUILDINGID~.,data = td_build,method = "knn",
                metric = "Accuracy",trControl = fitControl,preProcess = c("zv", "center", "scale"))


plot(knnFit)
knnFit


predict.knn <- predict(knnFit ,vd_new_2)
postResample(predict.knn , vd_new_2$BUILDINGID)
ConfusionMatrix<-confusionMatrix(vd_new_2$BUILDINGID , predict.knn)

ConfusionMatrix
save(knnFit, file = "knnFit.rda")
load("knnFit.rda")
rm(ConfusionMatrix)

#svm
SvmFit<-caret::train(BUILDINGID~., data= td_build, method="svmLinear", 
                            trControl=fitControl,preProcess= c("center", "scale"))

SvmFit
predict.svm <- predict(SvmFit ,vd_new_2)
predict.svm2 <- predict(SvmFit ,td_new_2)
postResample(predict.svm , vd_new_2$BUILDINGID)
confusionMatrix(vd_new_2$BUILDINGID , predict.svm)
rm(confusionMatrix)
confusionMatrix
save(SvmFit, file = "SvmFit.rda")
load("SvmFit.rda")
#add prediction column in training dataset 
td_new_2$build_prediction<-predict.svm2
vd_new_2$build_prediction<-predict.svm

td_new_2$B_fID<-as.factor(paste(td_new_2$BUILDINGID,td_new_2$FLOOR))
vd_new_2$B_fID<-as.factor(paste(vd_new_2$BUILDINGID,vd_new_2$FLOOR))
####FLOOR ####
td_floor<- select(td_new_2,-c(TIMESTAMP,USERID,PHONEID,max_2,count,SPACEID,BUILDINGID,FLOOR,
                             RELATIVEPOSITION,LATITUDE,LONGITUDE,max,max3,build_prediction))
#build.0 <- filter(td_floor, build_prediction == 0)
#build.1 <- filter(td_floor, build_prediction == 1)
#build.2 <- filter(td_floor, build_prediction== 2)

#vdbuild.0 <- filter(vd_new_2, build_prediction == 0)
#vdbuild.1 <- filter(vd_new_2, build_prediction == 1)
#vdbuild.2 <- filter(vd_new_2, build_prediction== 2)
#knn
fitControl <- trainControl(method = "repeatedcv", number=3,repeats = 3, 
                           verboseIter = TRUE, allowParallel = TRUE)
#build.0$FLOOR<- factor(build.0$FLOOR)
#vdbuild.0$FLOOR<-factor(vdbuild.0$FLOOR)
## knn
knnFit_floor <- train(B_fID~.,data = td_floor,method = "knn",
                metric = "Accuracy",trControl = fitControl,preProcess= c("center", "scale"))

knnFit_floor
#knnFit_floorb0

predict.knn_floor <- predict(knnFit_floor ,vd_new_2)
postResample(predict.knn_floor , vd_new_2$FLOOR)
ConfusionMatrix<-confusionMatrix(vd_new_2$FLOOR , predict.knn_floor)
ConfusionMatrix
save(knnFit_floor, file = "knnFit_floorb0.rda")
load("knnFit_floor.rda")
#svm
SvmFit_floor<-caret::train(B_fID~., data=td_floor, method="svmLinear", 
                     trControl=fitControl,preProcess= c("center", "scale"))
SvmFit_floor
predict.svm_floor <- predict(SvmFit_floor ,vd_new_2)
predict.svm_floor2 <- predict(SvmFit_floor ,td_new_2)
postResample(predict.svm_floor , vd_new_2$B_fID)
confusionMatrix(vd_new_2$B_fID , predict.svm_floor)
rm(confusionMatrix)
save(SvmFit_floor, file = "SvmFit_floor.rda")

td_new_2$floor_prediction<-predict.svm_floor2
vd_new_2$floor_prediction<-predict.svm_floor

# Add dummy variable for BuildingID&floor id 
DummyVar <- dummyVars("~BUILDINGID", data = td_new_2, fullRank=T)
DummyVarDF <- data.frame(predict(DummyVar, newdata = td_new_2))
td_new_2<-cbind(td_new_2, DummyVarDF)


DummyVar2 <- dummyVars("~BUILDINGID", data = vd_new_2, fullRank=T)
DummyVarDF2 <- data.frame(predict(DummyVar2, newdata = vd_new_2))
vd_new_2<-cbind(vd_new_2, DummyVarDF2)

#td_floor2<- select(td_new_2,-c(TIMESTAMP,USERID,PHONEID,max_2,count,SPACEID,BUILDINGID,
                             # RELATIVEPOSITION,LATITUDE,LONGITUDE,max,max3,build_prediction))
#Random Forest 
WAP_floor <- grep("WAP", names(td_floor), value=T)
bestmtry<-tuneRF(td_floor[WAP], td_floor$B_fID, ntreeTry=100, stepFactor=2, 
                 improve=0.05,trace=TRUE, plot=T)

system.time(RF_floor<-randomForest(B_fID~.,
                                 data= td_floor, 
                                 importance=T,maximize=T,
                                 method="rf", trControl=fitControl,
                                 ntree=100, mtry=52,allowParalel=TRUE))
save(RF_floor, file = "RF_floor.rda")
RF_floor
predict.rf_floor <- predict(RF_floor ,vd_new_2)
predict.rf_floor2 <- predict(RF_floor ,td_new_2)
postResample( predict.rf_floor , vd_new_2$B_fID)
CF_B_fID<-confusionMatrix(vd_new_2$B_fID , predict.rf_floor)
CF_B_fID
#add prediction column in training dataset 
td_new_2$floor_prediction<-predict.rf_floor2
vd_new_2$floor_prediction<-predict.rf_floor

####DUMMY FOR FLOOR ####
DummyVar3 <- dummyVars("~FLOOR", data = vd_new_2, fullRank=T)
DummyVarDF3 <- data.frame(predict(DummyVar3, newdata = vd_new_2))
vd_new_2<-cbind(vd_new_2, DummyVarDF3)

DummyVar4 <- dummyVars("~FLOOR", data = td_new_2, fullRank=T)
DummyVarDF4 <- data.frame(predict(DummyVar4, newdata = td_new_2))
td_new_2<-cbind(td_new_2, DummyVarDF4)

#dummy for predicted floors
DummyVar5 <- dummyVars("~floor_prediction", data = td_new_2, fullRank=T)
DummyVarDF5 <- data.frame(predict(DummyVar5, newdata = td_new_2))
td_new_2<-cbind(td_new_2, DummyVarDF5)

DummyVar6 <- dummyVars("~floor_prediction", data = vd_new_2, fullRank=T)
DummyVarDF6 <- data.frame(predict(DummyVar6, newdata = vd_new_2))
vd_new_2<-cbind(vd_new_2, DummyVarDF6)

#LONGITUTDE 
td_lon<- select(td_new_2,-c(TIMESTAMP,USERID,PHONEID,max_2,count,SPACEID,floor_prediction,
                              RELATIVEPOSITION,build_prediction,BUILDINGID,LATITUDE,max,max3,FLOOR,
                            FLOOR.1,FLOOR.2,FLOOR.3,FLOOR.4,B_fID))
#Random Forest 
WAP_lon <- grep("WAP", names(td_lon), value=T)
bestmtry<-tuneRF(td_lon[WAP_lon], td_lon$LONGITUDE, ntreeTry=100, stepFactor=2, 
                            improve=0.05,trace=TRUE, plot=T)
  
system.time(RF_lon<-randomForest(LONGITUDE~.,
                                      data= td_lon, 
                                      importance=T,maximize=T,
                                      method="rf", trControl=fitControl,
                                      ntree=100, mtry=52,allowParalel=TRUE))

save(RF_lon, file = "RF_lon.rda")
RF_lon

predict.rf_lon <- predict(RF_lon ,vd_new_2)
predict.rf_lon2 <- predict(RF_lon ,td_new_2)
postResample( predict.rf_lon , vd_new_2$LONGITUDE)
CF_LON<-confusionMatrix(vd_new_2$LONGITUDE , predict.rf_lon)


##SVM
build0lon <- filter(td_new_2, build_prediction == 0)
build0lon <- filter(td_new_2, build_prediction == 1)
build0lon <- filter(td_new_2, build_prediction== 2)

vdbuild0lon <- filter(vd_new_2, build_prediction == 0)
vdbuild1lon <- filter(vd_new_2, build_prediction == 1)
vdbuild2lon <- filter(vd_new_2, build_prediction== 2)

#SVM FOR BUILDING 0 
td_lon_b0<- select(build0lon,-c(TIMESTAMP,USERID,PHONEID,max_2,count,SPACEID,
                            RELATIVEPOSITION,BUILDINGID,LATITUDE,max,max3,FLOOR,
                            FLOOR.1,FLOOR.2,FLOOR.3,FLOOR.4,B_fID,floor_prediction,build_prediction))
#SVM 

SvmFit_lon_b0<-caret::train(LONGITUDE~., data= td_lon_b0, method="svmLinear", 
                           trControl=fitControl,preProcess= c("center", "scale"))
options(scipen=999)
SvmFit_lon_b0
predict.svm_lonb0 <- predict(SvmFit_lon_b0 ,vdbuild0lon)
predict.svm_lon_b02 <- predict(SvmFit_lon_bo ,build0lon)
postResample(predict.svm_lonb0 , vdbuild0lon$LONGITUDE)
save(SvmFit_lon, file = "SvmFit_lon_b0.rda")

####LATITUDE ####
td_lat<- select(td_new_2,-c(TIMESTAMP,USERID,PHONEID,max_2,count,SPACEID,floor_prediction,
                RELATIVEPOSITION,build_prediction,BUILDINGID,LONGITUDE,max,max3,FLOOR))
 #Random Forest 
WAP <- grep("WAP", names(td_lat), value=T)
bestmtry<-tuneRF(td_lat[WAP], td_lat$LONGITUDE, ntreeTry=100, stepFactor=2, 
                  improve=0.05,trace=TRUE, plot=T)
 
system.time(RF_lat<-randomForest(LATITUDE~.,
                                  data= td_lat, 
                                  importance=T,maximize=T,
                                  method="rf", trControl=fitControl,
                                  ntree=100, mtry=52,allowParalel=TRUE))
save(RF_lat, file = "RF_lat.rda")
RF_lat
predict.rf_lat <- predict(RF_lat ,vd_new_2)
postResample( predict.rf_lat , vd_new_2$LATITUDE)

####checking the errors of building prediction ###

td_err<- td_new_2 %>% select(WAP027,WAP028,SPACEID,USERID,PHONEID,
                             RELATIVEPOSITION,LATITUDE,LONGITUDE,FLOOR,BUILDINGID) %>%
  filter(apply(td_new_2[,1:2],1,function(x) any(x!=-105)))

ggplot() +
  geom_point(data = td_new_2, aes(x = LONGITUDE, y = LATITUDE, colour = "Training dataset")) +
  geom_point(data = vd_new_2, aes(x = LONGITUDE, y = LATITUDE, colour = "Test dataset")) +
  ggtitle("Locations (Training and Test sets)") 

#sampling te dataset 

td_lon_std<-td_lon
td_lon_std[,c(1:311)]<-td_lon_std[,c(1:311)] +105
td_lon_std<- td_lon_std%>%mutate_if(is.numeric,scale)
pca=princomp(td_lon_std[,c(1:311)], cor=TRUE)
summary(pca) 
pca$scores
biplot(pca)
str(td_lon[,300:326])
