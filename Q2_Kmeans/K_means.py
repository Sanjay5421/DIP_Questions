from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Loading image
image = imread(r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q2_Kmeans\image.png")
if image.dtype != np.uint8:
    image = (image * 255).astype(np.uint8)

# Reshape to Nx3
X = image.reshape(-1, 3)

# K-means clustering
kmeans = KMeans(n_clusters = 15, random_state=0, n_init=10)
labels = kmeans.fit_predict(X)

# Reconstruct image from centroids
centroids = kmeans.cluster_centers_.astype(np.uint8)
segmented_pixels = centroids[labels]
segmented_img = segmented_pixels.reshape(image.shape)

# Show and save
plt.imshow(segmented_img, cmap='gray')
plt.axis("off")
plt.show()
plt.imsave(r"C:\Users\USER\Desktop\College stuff\Sem 6\DIP\DIP_Questions\Q2_Kmeans\Output.jpg", segmented_img)
