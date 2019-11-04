
# IMPLEMENTACIÓN DE UNA RED NEURONAL ARTIFICIAL PARA LA CLASIFICACIÓN DE IMÁGENES DE SATÉLITE EN R
   


### REDES NEURONALES ARTIFICIALES EN INGENIERÍA HIDRÁULICA

MASTER UNIVERSITARIO EN INGENIERÍA HIDRÁULICA Y MEDIO AMBIENTE

Universitat Politècnica de València

Sergio Morell Monzo: sermonmo@doctor.upv.es




### RESUMEN

El siguiente texto describe el proceso seguido para realizar la clasificación de una imágen satelital Sentinel-2 entre dos tipos de coberturas: "suelo desnudo" y "cultivo". Para ello se ha utilizado una imagen Sentinel-2 del verano de 2018,  con código: S2B_MSIL1C_20180824T105019_N0206_R051_T30SYJ_20180824T151058. La imagen ha sido clasificada mediante la implementación de una Red Neuronal Artificial a través de la libreria "H2O" en lenguaje R. Las variables de entrada de la red han sido las bandas 2, 3, 4 y 8 de la imagen, todas ellas en una resolución de 10x10 metros.

El esquema de utilizado ha sido el siguiente:

<img src="images/esquema_proceso_rna_s2.png">

Figura 1: Esquema del proceso a seguir.

## 1. PREPROCESAMIENTO

CORRECCIÓN ATMOSFÉRICA

Para realizar la corrección atmosférica de la imagen se ha utilizado el software Sen2Cor 02.05.05 de la Agencia Espacial Europea. Sen2Cor se ha ejecutado a través de una linea de comandos en Windows para realizar la corrección atmosférica de la imagen nivel L1C a nivel L2A. La imagen con nivel de corrección L2A contiene los valores de Reflectancia de la superficie terrestre. El algoritmo utilizado realiza la corrección de cada una de las bandas de la imagen a todos los niveles de resolución posibles (60m, 20m y 10m). Las bandas con una resolución de 10m son las bandas B2, B3, B4 y B8 (Pancromática).


```R
C:\Users\User>cd C:\Sen2Cor-02.05.05-win64

C:\Sen2Cor-02.05.05-win64>L2A_Process.bat D:\S2B_MSIL1C_20180824T105019_N0206_R051_T30SYJ_20180824T151058.SAFE

pause
```

INSTALACIÓN DE LIBRERIAS

No todas las librerias son necesarias para realizar esta clasificación. Algunas de ellas solamente son necesarias para realizar la clasificación mediante otros clasificadores como Random Forest y Support Vector Machine peroel código para implementar estos clasificadores no se muestra en este trabajo. A continuación se muestran todas las librerias instaladas:


```R
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
```

DEFINICIÓN DE DIRECTORIO DE TRABAJO

Se ha creado una carpeta en D:/prac_ANN que contiene las cuatro bandas B2, B3, B4 y B8 de Sentinel-2 L2A en formato (.jp2). Estos archivos se obtienen después de realizar la corrección atmosférica en la ruta D:\ S2B_MSIL2A_20180824T105019_N0206_R051_T30SYJ_20180824T151058.SAFE\GRANULE\L2A_T30SYJ_A007656_20180824T105058\IMG_DATA\R10m


```R
# directorio de trabajo

setwd("D:/prac_ANN") 

getwd()
```

CONVERSIÓN DE LAS IMAGENES A FORMATO (.TIF) GEOTTIF

A continuación se convierten cada una de las bandas a formato .tif


```R
gdal_translate("banda02.jp2", "banda02_uncut.tif")

gdal_translate("banda03.jp2", "banda03_uncut.tif")

gdal_translate("banda04.jp2", "banda04_uncut.tif")

gdal_translate("banda08.jp2", "banda08_uncut.tif")


```

RECORTE DE LA ZONA DE ESTUDIO


```R
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
```

COMPOSICIÓN DEL RASTER MULTIBANDA


```R
# composicion raster multibanda

lay1 <- ("banda02.tif")
lay2 <- ("banda03.tif")
lay3 <- ("banda04.tif")
lay4 <- ("banda08.tif")

comp_mult_ST <- stack(lay1, lay2, lay3, lay4, lay5, lay6)

comp_mult_BR <- brick(comp_mult_ST)

writeRaster(comp_mult_BR, "comp_multi2348.tif", drivername="Gtiff")
```

CREACIÓN DEL DATASET (data.frame)


```R
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
```

En este caso no es necesario normalizar los datos ya que las variables B2, B3, B4 y B8 estan en niveles de reflectancia (%), que es una variable adimensional y todos los registros están entre el mismo rango de valores.

A pesar de ello podemos tipificar los datos restando la media y dividiendo entre la desviación estandar. Este proceso es realizado por la función "scale" de R:


```R
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
```

SEGMENTACIÓN DEL DATA SET

A continuación se realiza una partición del data set original (dfAll). El conjunto de datos se divide en 70% para entrenamiento (dfTrain) y 30% para validación (dfTest).


```R
set.seed(1234)

inBuild <- createDataPartition(y = dfAll$class, p = 0.7, list = FALSE)
dfTest <- dfAll[-inBuild,] # 30% test
dfTrain <- dfAll[inBuild,] # 70% test

summary(dfTest)

dummary(dfTrain)
```

BALANCEADO DE LOS DATOS DE ENTRENAMIENTO

Los algoritmos de machine learning presentan problemas cuando se enfrentan a conjuntos de datos desbalanceados. Si existe una gran diferencia en el número de muestras de cada categoria, puede que el clasificador no generalice bien para otros conjuntos de datos, generando modelos con overfitting o underfitting. Por ello es conveniente tener un número similar de muestras para cada categoria. En este caso se ha utilizado la técnica de submuestro, que consiste en eliminar datos de la categoría con menos muestras para compensar el número de muestras de ambas clases.


```R
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

```

## 2. CLASIFICACIÓN

ENTRENAMIENTO DEL MODELO: ANN


```R
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
```

PREDICCIÓN

A continuación se realiza la predicción para el Dataset de validación (dfTest):


```R
pred_ANN <- h2o.predict(modFit_ANN, newdata = as.h2o(dfTest))

pred_ANN
```

## 3. PERFORMANCE


```R
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
```

## 4. CLASIFICACIÓN DE LA IMAGEN

Finalmente se realiza la clasificación de la imagen raster según los dos tipos de coberturas (suelo desnudo / cultivaado) y se guarda el raster generado en el disco:


```R
# CLAIFICACION DEL AREA DE ESTUDIO

# clasificacion de la imagen
beginCluster()
system.time(raster_ANN <- clusterR(img, raster::predict, args= list(model = h2o.getFrame(modFit_ANN)))) #!
endCluster()

plot(raster_ANN)

# guardar imagen en disco
writeRaster(raster_ANN,"raster_ANNpred.tiff", drivername="Gtiff")
```


```R
pause
```
