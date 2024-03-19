import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fungsi untuk preprocess dan prediksi gambar
def predict_image(image):
    image = image.resize((150, 150))
    image = image.convert('RGB')
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    class_indices = {'Alpukat': 0, 'Apel': 1, 'Pisang': 2}  # Update class indices accordingly
    class_names = sorted(class_indices, key=class_indices.get)
    predicted_label = class_names[predicted_class]
    return predicted_label

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Pilih Halaman", ["Home", "Tentang Project", "Take Picture", "Klasifikasi"])

#Main Page
if(app_mode == "Home"):
    # st.header("Mengklasifikasikan buah Apel, Alpukat, dan Pisang")
    st.write("<h1 style='text-align: center;'>Mengklasifikasikan buah Apel, Alpukat, dan Pisang</h1>", unsafe_allow_html=True)
    st.write("<hr>", unsafe_allow_html=True)
    st.write("")
    st.write("<div style='text-align: center;'><span style='font-size: 35px; font-weight: bold;'>Kelompok 4</span></div>", unsafe_allow_html=True)
    st.write("")
    st.write("<div style='text-align: center;'>Ahmad Dhiya Ulhaqi - 2109106056</div>", unsafe_allow_html=True)
    st.write("")
    st.write("<div style='text-align: center;'>Muhammad Irvan Hakim - 2109106057</div>", unsafe_allow_html=True)
    st.write("")
    st.write("<div style='text-align: center;'>M.Dhimas Eko Wiyono - 2109106068</div>", unsafe_allow_html=True)
    st.write("")
    st.write("<div style='text-align: center;'>Pranata Eka Pramudya - 2109106077</div>", unsafe_allow_html=True)
    st.write("")
 
#Tentang Project
elif(app_mode == "Tentang Project"):
    st.write("")
    st.write("<h1 style='text-align: center;'>Tentang Project</h1>", unsafe_allow_html=True)
    st.subheader("Tujuan Project")
    st.write("Tujuan dari proyek ini adalah untuk mengembangkan sebuah sistem pengklasifikasi buah yang dapat mengidentifikasi dan membedakan antara jenis buah Apel, Alpukat, dan Pisang dengan menggunakan teknologi Computer Vison dan Machine Learning.\n")

#Take Picture
elif(app_mode == "Take Picture"):
    st.write("<h1 style='text-align: center;'>Ambil Gambar</h1>", unsafe_allow_html=True)
    
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Untuk membaca image file buffer sebagai PIL Image:
        img = Image.open(img_file_buffer)

        # Mengubah PIL Image menjadi numpy array:
        img_array = np.array(img)

        # Display gambar yang diambil
        st.image(img, width=400, caption='Gambar yang diambil')

        # Memuat model TensorFlow
        model = tf.keras.models.load_model('model-bw.h5')

        # Prediksi gambar yang diambil 
        predicted_label = predict_image(img)

        # Display prediksi
        st.write("Hasil Klasifikasi: ", predicted_label)


elif(app_mode == "Klasifikasi"):
    st.write("<h1 style='text-align: center;'>Klasifikasi Gambar</h1>", unsafe_allow_html=True)
    test_image = st.file_uploader("Pilih Gambar:")
    
    # Memuat model TensorFlow
    model = tf.keras.models.load_model('model-bw.h5')

    #Checkbox untuk mengontrol tampilan gambar
    show_image = st.checkbox("Tampilkan gambar")
    
    if show_image and test_image is not None:
        st.image(test_image, width=400, caption='Gambar yang dipilih')
    elif show_image and test_image is None:
        st.write("Anda belum memilih gambar")

    #Tombol Klasifikasi
    if(st.button("Klasifikasi") and test_image is not None):
        # Load dan preprocess gambar yang di upload
        image = Image.open(test_image)
        predicted_label = predict_image(image)

        # Display prediksi
        st.write("Hasil Klasifikasi: ", predicted_label)