
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


set.seed(1234)

inBuild <- createDataPartition(y = dfAll$class, p = 0.7, list = FALSE)
dfTest <- dfAll[-inBuild,] # 30% test
dfTrain <- dfAll[inBuild,] # 70% test

summary(dfTest)

dummary(dfTrain)

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


set.seed(1234)



