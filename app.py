import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie


# --- configuraties ---
st.set_page_config(page_title="Kleuren Palet Generator", page_icon=":spades:", layout="wide")


# --- lottie functie ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_ig53qvih.json")


# --- Session state ----
if "Image" not in st.session_state:
    st.session_state["Image"]="not done"

def change_image_state():
    st.session_state["Image"]="done"

# --- Tekst en afbeelding (lottie) ---
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("# Kleuren Palet Generator")
        st.markdown(" ")
        st.markdown("Deze kleurenpalet generator is een tool die je kan gebruiken om kleuren te selecteren uit een afbeelding. Het werkt door een afbeelding te uploaden en vervolgens analyseert het programma de afbeelding om de meest voorkomende kleuren te identificeren. Deze kleuren worden vervolgens gepresenteerd als een kleurenpalet dat je kan gebruiken voor jouw ontwerpen.")
        st.markdown(" ")
    with right_column:
        st_lottie(lottie_coding, height= 300)


tab1, tab2 = st.tabs(["Kleur Palet Generator", "Uitleg Code"])
with tab1:
        

    # --- File Uploader ---
    left_column2, mid_column2, right_column2 = st.columns([1,4,1])
    with mid_column2:
        img_file = st.file_uploader("Upload afbeelding", key="file_uploader", on_change=change_image_state)
        if img_file is not None:
            try:
                img = Image.open(img_file)
            except:
                st.error("Dit bestand kon niet geupload worden, upload een jpg of png bestand")
        st.markdown(" ")


    # --- Show afbeelding ---
    if st.session_state["Image"] == "done":
        with mid_column2:
            st.image(img)

        st.markdown("---")

        # --- RGB extraheren uit afbeelding ----
        sample_size = 1000
        r, g, b = np.array(img).reshape(-1,3).T
        df_rgb = pd.DataFrame({"R": r, "G": g, "B": b}).sample(n=sample_size)

        st.markdown(" ")


        # --- Palet grote selectie ---
        left_column3, mid_column3, right_column3 = st.columns([1,4,1])
        with mid_column3:
            pallet_size = st.slider(
                "Selecteer palet grote",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Het aantal kleuren vanuit de afbeelding"
            )

        st.markdown(" ")
        # --- KMeans model ---
        model = KMeans(n_clusters=pallet_size)
        clusters = model.fit_predict(df_rgb)

        palette = model.cluster_centers_.astype(int).tolist()


        # --- RGB naar Hex ---
        dfcolor = pd.DataFrame(palette, columns=["R","G","B"])

        def rgb_to_hex(red, green, blue):
            #Return color as #rrggbb for the given color values.
            return '#%02x%02x%02x' % (red, green, blue)

        dfcolor['hex'] = dfcolor.apply(lambda r: rgb_to_hex(*r), axis=1)


        # --- Hex naar color picker ---
        columns = st.columns(pallet_size)
        for i, col in enumerate(columns):
            with col:        
                st.session_state[f"col_{i}"]= \
                    st.color_picker(label=str(i+1), 
                    value=dfcolor['hex'][i], 
                    key=f"pal_{i}")

        for i, col in enumerate(columns):
            with col:
                st.markdown("Hex code:")
                st.code(dfcolor['hex'][i])

with tab2:
    st.markdown("## Uitleg code")
    st.markdown(" ")
    st.markdown("Voor de palet genator is machine learning gebruikt. Het algoritme KMean van scikit-learn. Als eerste worden de RGB waardes uit de afbeelding gehaald. Vervolgens is er een iteratief proces waarbij punten in een dataset worden toegewezen aan een beperkt aantal klusters op basis van hun afstand tot de klustercenter. In dit geval zijn het aantal clusters gelijk aan het aantal kleuren.")
    st.markdown(" ")
    st.markdown("Importeer de juiste libraries.")
    st.code('''import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie''', language='python')

    st.markdown(" ")
    st.markdown("Stel de configuraties in.")
    st.code('''st.set_page_config(page_title="Kleuren Palet Generator", page_icon=":spades:", layout="wide")''', language='python')

    st.markdown(" ")
    st.markdown("De functie om de lottie afbeelding te kunnen laden.")
    st.code('''def load_lottieurl(url): 
        r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
    
lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_ig53qvih.json")''', language='python')

    st.markdown(" ")
    st.markdown("De session state functie om later alleen bepaalde onderdelen van de code te laden als de afbeelding succesvol is ingeladen.")
    st.code('''if "Image" not in st.session_state:
    st.session_state["Image"]="not done"

def change_image_state():
    st.session_state["Image"]="done"''',language='python')

    st.markdown(" ")
    st.markdown("De uitleg en afbeelding verdeeld over twee kolommen.")
    st.code('''with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("# Kleuren Palet Generator")
        st.markdown(" ")
        st.markdown("Stuk tekst met uitleg.")
        st.markdown(" ")
    with right_column:
        st_lottie(lottie_coding, height= 300)''',language='python')
    
    st.markdown(" ")
    st.markdown("De file uploader, met drie kolommen waarvan alleen de middelste gebruikt wordt.")
    st.code('''left_column2, mid_column2, right_column2 = st.columns([1,4,1])
    with mid_column2:
        img_file = st.file_uploader("Upload afbeelding", key="file_uploader", on_change=change_image_state)
        if img_file is not None:
            try:
                img = Image.open(img_file)
            except:
                st.error("Dit bestand kon niet geupload worden, upload een jpg of png bestand")''',language='python')
    
    st.markdown(" ")
    st.markdown("Zodra de afbeelding is geupload zal die worden weergegeven.")
    st.code('''if st.session_state["Image"] == "done":
    with mid_column2:
        st.image(img)''',language='python')
    
    st.markdown(" ")
    st.markdown("Dit is de slider waarmee het aantal kleuren geselecteerd wordt.")
    st.code('''left_column3, mid_column3, right_column3 = st.columns([1,4,1])
    with mid_column3:
        pallet_size = st.slider(
            "Selecteer palet grote",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Het aantal kleuren vanuit de afbeelding"
        )''',language='python')
    
    st.markdown(" ")
    st.markdown("Hier worden de RGB waardes uit de afbeelding gehaald. Ook wordt het KMeans model toegepast, waar een aantal RGB codes uitkomen. Deze worden vervolgens omgezet naar een HEX code.")
    st.code('''sample_size = 1000
    r, g, b = np.array(img).reshape(-1,3).T
    df_rgb = pd.DataFrame({"R": r, "G": g, "B": b}).sample(n=sample_size)

model = KMeans(n_clusters=pallet_size)
clusters = model.fit_predict(df_rgb)

palette = model.cluster_centers_.astype(int).tolist()

dfcolor = pd.DataFrame(palette, columns=["R","G","B"])

def rgb_to_hex(red, green, blue):
    #Return color as #rrggbb for the given color values.
    return '#%02x%02x%02x' % (red, green, blue)

dfcolor['hex'] = dfcolor.apply(lambda r: rgb_to_hex(*r), axis=1)''',language='python')

    st.markdown(" ")
    st.markdown("Als laatste worden de HEX codes omgezet naar een color picker, die elk in zijn eigen kolom staat.")
    st.code('''columns = st.columns(pallet_size)
    for i, col in enumerate(columns):
        with col:        
            st.session_state[f"col_{i}"]= \
                st.color_picker(label=str(i+1), 
                value=dfcolor['hex'][i], 
                key=f"pal_{i}")

for i, col in enumerate(columns):
    with col:
        st.write(dfcolor['hex'][i])''',language='python')