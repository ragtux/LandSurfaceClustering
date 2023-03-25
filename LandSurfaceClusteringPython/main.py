# This script assumes the following directory structure:
# ├──├── LC09_L1TP_031026_20230103_20230315_02_T1
# │  ├── LC09_L1TP_031026_20230103_20230315_02_T1_ANG.txt
# │  ├── LC09_L1TP_031026_20230103_20230315_02_T1_B1.TIF
# │  ├── LC09_L1TP_031026_20230103_20230315_02_T1_B2.TIF
#    ...        ...         ...
# │  ├── LC09_L1TP_031026_20230103_20230315_02_T1_VAA.TIF
# │  └── LC09_L1TP_031026_20230103_20230315_02_T1_VZA.TIF
# ├── LC09_L1TP_031026_20230308_20230308_02_T2
# │  ├── LC09_L1TP_031026_20230308_20230308_02_T2_ANG.txt
# │  ├── LC09_L1TP_031026_20230308_20230308_02_T2_B1.TIF
# │  ├── LC09_L1TP_031026_20230308_20230308_02_T2_B2.TIF
# │  ├── LC09_L1TP_031026_20230308_20230308_02_T2_B3.TIF
# ... ... ...
# │  ├── LC09_L1TP_031026_20230308_20230308_02_T2_VAA.TIF
# │  └── LC09_L1TP_031026_20230308_20230308_02_T2_VZA.TIF
#
#

from pyspark.sql.functions import *
import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import cv2
from pyspark.sql import SparkSession
import os

from PIL import Image, ImageDraw


# WALKTHROUGH
# Dataframe column:
data_dir = '/mnt/raid/se513_midterm'
df_columns = ['scene_id', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9']

data = []
pandas_image_df = pd.DataFrame(columns=df_columns)

# spark.read.format("image").load("train_set/dir_0/*.jpg", "train_set/dir_2/*.png").selectExpr("image", "image.origin as label").show

spark = SparkSession.builder.appName("LandSatExample") \
    .config("spark.driver.memory", "20g") \
    .config("spark.driver.maxResultSize", "20g") \
    .config("executor-memory", "60g") \
    .getOrCreate()

# (1) load remote sensing data into Spark pipeline...
# struct ['origin', 'height', 'width', 'nChannels', 'mode', 'data']
df = spark.read.format("image") \
    .option("pathGlobFilter", "*.TIF")\
    .option("recursiveFileLookup", "true")\
    .load("/mnt/raid/mini_midterm")

df.createOrReplaceTempView("scenes")

# df = spark.sql("SELECT origin, height, width, nChannels, mode, data FROM scenes")

sql01 = """
SELECT 
    split_part(image.origin,'/', -2) as Scene, 
    split_part(split_part(image.origin, '_', -1),'.',1) as Band,
    image.data as Data
FROM scenes
"""

df2 = spark.sql(sql01)

df_pivot = df2.groupBy("Scene").pivot("Band").agg(first("Data"))




df_pivot.printSchema()
df_pivot.show(truncate=True)

image_row = 40
spark_single_img = df_pivot.select("image").collect()[image_row]
(spark_single_img.image.origin, spark_single_img.image.mode, spark_single_img.image.nChannels )

mode = 'RGBA' if (spark_single_img.image.nChannels == 4) else 'RGB'
Image.frombytes(mode=mode, data=bytes(spark_single_img.image.data), size=[spark_single_img.image.width,spark_single_img.image.height]).show()



df2.write.parquet("./scenes.parquet")

# df2_piv = df2.groupBy("SceneName").pivot("Band")

# df2_piv.show()


df.selectExpr("*").show()

# for each of the top level scene folders...
band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9']
Nsamples = 2000  # number of random samples used to "train" k-means here
NUMBER_OF_CLUSTERS = 4  # the number of independent clusters for k-means
colour_map = 'terrain'  # cmap, see matplotlib.org/examples/color/colormaps_reference.html

# (1) load remote sensing data into Spark pipeline...
image_df = spark.read.format("image").load('/mnt/raid/mini_midterm/LC08_L1GT_025027_20211010_20211019_02_T2/*.TIF')

# pivot image_df
image_df = image_df.select("image.origin", "image.data")
image_df = image_df.withColumn("image.data", image_df["data"].cast("array<float>"))


image_df.select("image.origin", "image.height", "image.width", "image.mode", "image.nChannels").show(5, truncate=False)


# image_df.select("image.origin", "image.height", "image.width", "image.mode", "image.nChannels").show(5, truncate=False)

images = dict()
for image_path in glob.glob(image_folder_name + '/**/*.TIF'):
    print('reading ', image_path)
    # Read in the image
    landsat_tif_image = cv2.imread(image_path)
    # Change color to RGB (from BGR)
    temp = cv2.cvtColor(landsat_tif_image, cv2.COLOR_BGR2RGB)
    #    temp = imageio.v2.imread(image_path)
    temp = temp[:, :, 0].squeeze()
    sindex = image_path.rindex('_')
    eindex = image_path.rindex('.')
    print(image_path[sindex + 1:eindex])
    images[image_path[sindex + 1:eindex]] = temp  # FOR DIFFERENT FILE NAMES, ADJUST THIS!
#    plt.imshow(image)
#    plt.show()

print('images have ', np.size(temp), ' pixels each')
xj = images.keys()
print(xj)


# make a 3D numpy array of data...
imagecube = np.zeros([images['B2'].shape[0], images['B2'].shape[1], np.size(band_names)])
for j in np.arange(np.size(band_names)):
    imagecube[:, :, j] = images[band_names[j]]  #
imagecube = imagecube / 256  # scaling to between 0 and 1


# display an RGB or false colour image
thefigsize = (10, 8)  # set figure size
# plt.figure(figsize=thefigsize)
# plt.imshow(imagecube[:,:,0:3])

# sample random subset of images
imagesamples = []
for i in range(Nsamples):
    xr = np.random.randint(0, imagecube.shape[1] - 1)
    yr = np.random.randint(0, imagecube.shape[0] - 1)
    imagesamples.append(imagecube[yr, xr, :])
# convert to pandas dataframe
imagessamplesDF = pd.DataFrame(imagesamples, columns=band_names)

# make pairs plot (each band vs. each band)
seaborn_params_p = {'alpha': 0.15, 's': 20, 'edgecolor': 'k'}
# pp1=sns.pairplot(imagessamplesDF, plot_kws = seaborn_params_p)#, hist_kws=seaborn_params_h)

# fit kmeans to samples:
from sklearn.cluster import KMeans

KMmodel = KMeans(n_clusters=NUMBER_OF_CLUSTERS)
KMmodel.fit(imagessamplesDF)
KM_train = list(KMmodel.predict(imagessamplesDF))
i = 0
for k in KM_train:
    KM_train[i] = str(k)
    i = i + 1
imagessamplesDF2 = imagessamplesDF
imagessamplesDF2['group'] = KM_train
# pair plots with clusters coloured:
pp2 = sns.pairplot(imagessamplesDF, vars=band_names, hue='group', plot_kws=seaborn_params_p)
pp2._legend.remove()

#  make the clustered image
imageclustered = np.empty((imagecube.shape[0], imagecube.shape[1]))
i = 0
for row in imagecube:
    temp = KMmodel.predict(row)
    imageclustered[i, :] = temp
    i = i + 1
# plot the map of the clustered data
plt.figure(figsize=thefigsize)
plt.imsave('LC08_L1GT_028029_20191229_20200824_02_T2.png', imageclustered, cmap=colour_map)
# plt.imshow(imageclustered, cmap=colour_map)
# plt.show()
