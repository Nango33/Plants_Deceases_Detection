import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Acceuil","À propos","Reconnaissance des Maladies"])

#Main Page
if(app_mode=="Acceuil"):
    st.header("SYSTÈME DE RECONNAISSANCE DES MALADIES DES PLANTES")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    
Bienvenue dans le Système de Reconnaissance des Maladies des Plantes ! 🌿🔍
    
Notre mission est d'aider à identifier les maladies des plantes de manière efficace. Téléchargez une image d'une plante, et notre système l'analysera pour détecter d'éventuels signes de maladies. Ensemble, protégeons nos cultures et assurons une récolte plus saine !

### Comment ça marche ?
1. **Téléchargez l'image:** Rendez-vous sur la page de**Disease Recognition** et téléchargez une image d'une plante présentant des symptômes suspects.
2. **Analyse:** Notre système traitera l'image à l'aide d'algorithmes avancés pour identifier d'éventuelles maladies.
3. **Résultats:** Consultez les résultats et les recommandations pour les actions à entreprendre.

### Pourquoi nous choisir ?
- **Précision:** Notre système utilise des techniques d'apprentissage automatique de pointe pour une détection précise des maladies.
- **Facilité d'utilisation:** Interface simple et intuitive pour une expérience utilisateur fluide.
- **Rapide et efficace :** Recevez les résultats en quelques secondes, permettant une prise de décision rapide.

### Commencez !
Cliquez sur la page de **Disease Recognition** dans la barre latérale pour télécharger une image et découvrir la puissance de notre Système de Reconnaissance des Maladies des Plantes!

""")

#À propos
elif(app_mode=="À propos"):
    st.header("À propos")
    st.markdown("""
                #### À propos du jeu de données
                Ce jeu de données a été recréé à l'aide d'une augmentation hors ligne à partir du jeu de données original. Le jeu de données original peut être trouvé sur ce dépôt GitHub.
                Ce jeu de données se compose d'environ 87 000 images RGB de feuilles de cultures saines et malades, classées en 38 catégories différentes. L'ensemble total du jeu de données est divisé selon un ratio de 80/20 entre l'ensemble d'entraînement et l'ensemble de validation, tout en préservant la structure des répertoires.
                Un nouveau répertoire contenant 33 images de test a ensuite été créé à des fins de prédiction.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Reconnaissance des Maladies"):
    st.header("Reconnaissance des Maladies")
    test_image = st.file_uploader("Choisissez une image:")
    if(st.button("Afficher l'image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Prédire")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))