import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import cv2
from pyspark.sql import SparkSession
import os


# WALKTHROUGH
# Dataframe column:
data_dir = '/mnt/raid/se513_midterm'
df_columns = ['scene_id', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9']

data = []

pandas_image_df = pd.DataFrame(columns=df_columns)

for scene_folder in os.listdir(data_dir):
    row = [None] * 9 # create a None list for each row
    row[0] = scene_folder
    scene_folder_full_path = os.path.join(data_dir, scene_folder)
    if os.path.isdir(scene_folder_full_path):
        print(scene_folder_full_path)
        for file in os.listdir(scene_folder_full_path):
            if file.endswith('.TIF'):
                g = file.split('_')[-1].split('.')[0]
                # LC08_L1GT_025027_20180612_20200831_02_T2_B2.TIF <-- above line gets the 'B2' portion of the file name.
                # ...if starts with 'B' than we know it's a band name
                if g.startswith('B'):
                    try:
                        index = df_columns.index(g) # get the index of the column name
                        row[index] = os.path.join(scene_folder_full_path, file)
                    except Exception as e:
                        print("- - - -> {0}".format(e))
                    g = g[1:]
        pandas_image_df.loc[len(pandas_image_df)] = row


spark = SparkSession.builder.appName("LandSatExample") \
    .config("spark.driver.memory", "20g") \
    .config("spark.driver.maxResultSize", "20g") \
    .config("executor-memory", "60g") \
    .getOrCreate()

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


# image_df.show()
# images_dir = "/mnt/raid/mini_midterm"
# image_df.select("image.origin", "image.height", "image.width", "image.mode", "image.nChannels").show(5, truncate=False)

# import images to dictionary:
images = dict()
for image_path in glob.glob(image_folder_name + '/**/*.TIF'):
    print('reading ', image_path)
    # Read in the image
    image = cv2.imread(image_path)
    # Change color to RGB (from BGR)
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
