# Image search engine on SIFT + K-Means + Histogram
For start put two image collectiona on ./data
Get image descriptors
```
descriptors = get_descriptors('./webapp/data/train/Images')
```
Fit K_mean_model
```
K_Means_model = KMeans(n_clusters=1024, random_state=0).fit(descriptors)
```
Create indexed image database
```
df = create_database('./webapp/data/test/Images', K_Means_model)
df.to_csv('database1024.csv', index=False)
```
Fit NN_model
```
NN_model  = NearestNeighbors(metric='cosine').fit(vectors)
```
Now you can offload model weights and run web interface by
```
streamlit run streamlit.py
```

![image](https://github.com/Shireee/cv-labs/assets/52496230/5b3cc11f-454c-4512-8638-a1b6a05a8cd7)
