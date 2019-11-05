
C:\Users\User>cd C:\Sen2Cor-02.05.05-win64

C:\Sen2Cor-02.05.05-win64>L2A_Process.bat D:\S2B_MSIL1C_20180824T105019_N0206_R051_T30SYJ_20180824T151058.SAFE

pause

# instalacion de librerias

install.packages("sp")
install.packages("raster")
install.packages("rgdal")
install.packages("gdalUtils")
install.packages("caret")
install.packages("snow")
install.packages("e1071")
install.packages("h2o")

# carga de librerias

library(sp)
library(raster)
library(rgdal)
library(gdalUtils)
library(caret)
library(snow)
library(e1071)
library(h2o)

# directorio de trabajo

setwd("D:/prac_ANN") 

getwd()

gdal_translate("banda02.jp2", "banda02_uncut.tif")

gdal_translate("banda03.jp2", "banda03_uncut.tif")

gdal_translate("banda04.jp2", "banda04_uncut.tif")

gdal_translate("banda08.jp2", "banda08_uncut.tif")



# zona de estudio

zona_estudio <- readOGR("D:/prac_ANN/zona_estudio/zona_estudio.shp")

banda02_uncut <- raster("banda02_uncut.tif")
banda02 <- crop(banda02_uncut, zona_estudio)
writeRaster(banda02, "banda02.tif", drivername="Gtiff")

banda03_uncut <- raster("banda03_uncut.tif")
banda03 <- crop(banda03_uncut, zona_estudio)
writeRaster(banda03, "banda03.tif", drivername="Gtiff")

banda04_uncut <- raster("banda04_uncut.tif")
banda04 <- crop(banda04_uncut, zona_estudio)
writeRaster(banda04, "banda04.tif", drivername="Gtiff")

banda08_uncut <- raster("banda08_uncut.tif")
banda08 <- crop(banda08_uncut, zona_estudio)
writeRaster(banda08, "banda08.tif", drivername="Gtiff")

# composicion raster multibanda

lay1 <- ("banda02.tif")
lay2 <- ("banda03.tif")
lay3 <- ("banda04.tif")
lay4 <- ("banda08.tif")

comp_mult_ST <- stack(lay1, lay2, lay3, lay4, lay5, lay6)

comp_mult_BR <- brick(comp_mult_ST)

writeRaster(comp_mult_BR, "comp_multi2348.tif", drivername="Gtiff")

# raster multibanda

img <- brick("comp_multi2348.tif")

img

# rois de entrenamiento

trainData <- shapefile("D:/s2_img/datos_entrenamiento/datos_entrenamiento.shp")
responseCol <- "class" #el shapefile debe contener un unico campo con nombre "class"

trainData

# extraccion de valores de pixel en las areas de entrenamiento

dfAll = data.frame(matrix(vector(), nrow = 0, ncol = length(names(img)) + 1))   
for (i in 1:length(unique(trainData[[responseCol]]))){
  category <- unique(trainData[[responseCol]])[i]
  categorymap <- trainData[trainData[[responseCol]] == category,]
  dataSet <- extract(img, categorymap)
  if(is(trainData, "SpatialPointsDataFrame")){
    dataSet <- cbind(dataSet, class = as.numeric(rep(category, nrow(dataSet))))
    dfAll <- rbind(dfAll, dataSet[complete.cases(dataSet),])
  }
  if(is(trainData, "SpatialPolygonsDataFrame")){
    dataSet <- dataSet[!unlist(lapply(dataSet, is.null))]
    dataSet <- lapply(dataSet, function(x){cbind(x, class = as.numeric(rep(category, nrow(x))))})
    df <- do.call("rbind", dataSet)
    dfAll <- rbind(dfAll, df)
  }
}

# nombre de variables
names(dfAll) = c("B2", "B3", "B4", "B8", "class")

dfAll

# conversion variable 'class' a tipo factor
class(dfAll$class)
dfAll$class <- factor(dfAll$class, levels = c("0", "1"), labels = c("desnudo", "cultivo"))
class(dfAll$class)

# NORMALIZACION DE LOS DATOS
library(ggplot2)

#b2
qqnorm(dfAll$B2, main= "B2") # Distribucion de la variable B2 frente a su distribucion teorica normal
qqline(dfAll$B2)

#b3
qqnorm(dfAll$B3, main= "B3") # Distribucion de la variable B3 frente a su distribucion teorica normal
qqline(dfAll$B3)

#b4
qqnorm(dfAll$B4, main= "B4") # Distribucion de la variable B4 frente a su distribucion teorica normal
qqline(dfAll$B4)

#b8
qqnorm(dfAll$B8, main= "B8") # Distribucion de la variable B8 frente a su distribucion teorica normal
qqline(dfAll$B8)

# normalizar (funcion scale)
dfAll[, c(1:4)] <- scale(dfAll[, c(1:4)])

#b2
qqnorm(dfAll$B2, main= "B2") # Distribucion de la variable B2 frente a su distribucion teorica normal
qqline(dfAll$B2)

#b3
qqnorm(dfAll$B3, main= "B3") # Distribucion de la variable B3 frente a su distribucion teorica normal
qqline(dfAll$B3)

#b4
qqnorm(dfAll$B4, main= "B4") # Distribucion de la variable B4 frente a su distribucion teorica normal
qqline(dfAll$B4)

#b8
qqnorm(dfAll$B8, main= "B8") # Distribucion de la variable B8 frente a su distribucion teorica normal
qqline(dfAll$B8)

# hist
hist(dfAll$B2, main= "B2", col= "blue")
hist(dfAll$B3, main= "B3", col= "green")
hist(dfAll$B4, main= "B4", col= "red")
hist(dfAll$B8, main= "B8", col= "grey")

set.seed(1234)

inBuild <- createDataPartition(y = dfAll$class, p = 0.7, list = FALSE)
dfTest <- dfAll[-inBuild,] # 30% test
dfTrain <- dfAll[inBuild,] # 70% test

summary(dfTest)

dummary(dfTrain)

# BALANCEADO DEL DATASET

# funcion undersampling_df
undersample_ds <- function(x, classCol, nsamples_class){
  for (i in 1:length(unique(x[, classCol]))){
    class.i <- unique(x[, classCol])[i]
    if((sum(x[, classCol] == class.i) - nsamples_class) != 0){
      x <- x[-sample(which(x[, classCol] == class.i), 
                     sum(x[, classCol] == class.i) - nsamples_class), ]
    }
  }
  return(x)
}

# recuento de muestras por categoria
desnudo <- sum(dfTrain$class == "desnudo") # 0 = desnudo
cultivo <- sum(dfTrain$class == "cultivo") # 1 = cultivo

desnudo
cultivo

#graf
pie_values <- c(desnudo, cultivo)
pie_labels <- c("desnudo", "cultivo")

pie(pie_values, pie_labels)

# recorte del dataset
sumatorio_clases <- c(desnudo, cultivo)

nsamples_class <- min(sumatorio_clases)

dfTrain_bal <- undersample_ds(dfTrain, "class", nsamples_class)

# comprobacion
desnudo_bal <- sum(dfTrain_bal$class == "desnudo") # 0 = desnud
cultivo_bal <- sum(dfTrain_bal$class == "cultivo") # 1 = cultivo

desnudo_bal
cultivo_bal

#graf
pie_values_bal <- c(desnudo_bal, cultivo_bal)
pie_labels_bal <- c("desnudo", "cultivo")

pie(pie_values_bal, pie_labels_bal)


set.seed(234)

# ENTRENAMIENTO DEL MODELO

# procesamiento en paralelo
h2o.init(nthreads = -1)

# red neuronal
modFit_ANN = h2o.deeplearning(y = 'class',
                              training_frame = as.h2o(dfTrain_bal),
                              activation = 'Rectifier',
                              hidden = c(3, 3),
                              loss = "CrossEntropy"
                              epochs = 80,
                              rate = 0,03,
                              train_samples_per_iteration = -2)

# Default parameters:

# retro-propagacion del error: Classic Backpropagation
# momentum: 0
# psp: RBF

pred_ANN <- h2o.predict(modFit_ANN, newdata = as.h2o(dfTest))

pred_ANN

# EVALUACION DEL MODELO

# matriz de confusion
yANN_pred <- as.vector(ifelse(pred_ANN$predict == 'desnudo', 0, 1))
yANN_dfTest <- ifelse(dfTest$class == 'desnudo', 0, 1)

conf_matrix <- table(yANN_dfTest, yANN_pred)

conf_matrix

# precision
h2o.performance(modFit_ANN)

# curva ROC
pred1_ANN <- prediction(as.numeric(yANN_pred), as.numeric(yANN_dfTest))

perf1_ANN <- performance(pred1_ANN, "tpr", "fpr")

plot(perf1_ANN)

# CLAIFICACION DEL AREA DE ESTUDIO

# clasificacion de la imagen
beginCluster()
system.time(raster_ANN <- clusterR(img, raster::predict, args= list(model = h2o.getFrame(modFit_ANN)))) #!
endCluster()

plot(raster_ANN)

# guardar imagen en disco
writeRaster(raster_ANN,"raster_ANNpred.tiff", drivername="Gtiff")

#pause
