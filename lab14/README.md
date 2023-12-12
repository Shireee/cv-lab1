# Image search engine with CLIP (Contrastive Language-Image Pre-Training) model 
For start put image collection on ./data

Load the model
```
model, transform = clip.load('ViT-B/32', device=device)
```
Create indexed image database
```
df = create_database('./webapp/data/voc2012/Images', model)
df.to_csv('./webapp/data/database.csv', index=False)
```
Fit NN_model and offload model 
```
NN_model  = NearestNeighbors(metric='cosine').fit(np.array(df['vector'].apply(ast.literal_eval).tolist()))

with open('./webapp/models/NN_model.pkl', 'rb') as f:
    NN_model = pickle.load(f)
```
Now you can run web interface 
```
streamlit run streamlit.py
```

![image](https://github.com/Shireee/cv-labs/assets/52496230/d284723c-a5a6-4d5f-a7b6-8e0bcd344195)
