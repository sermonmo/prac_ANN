
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

# extracciON de valores de pixel en las areas de entrenamiento

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

```

## 2. CLASIFICACIÓN

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
# funnción undersample_df

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

desnudo <- sum(dfTrain$class == desnudo)
desnudo

cultivo <- sum(dfTrain$class == cultivo)
cultivo

pie(desnudo, cultivo)

# recorte de las muestras

nsamples_class <- 100

dfTrain_bal <- undersample_ds(dfTrain, "class", nsamples_class)

```

ENTRENAMIENTO DEL MODELO: ANN


```R
set.seed(1234)



```
