import streamlit as st
import cv2
import numpy as np
import face_recognition
import os

# Función para capturar el rostro y guardarlo
def capture_face():
    cap = cv2.VideoCapture(0)
    st.write("Capturando rostro... Presiona 's' para guardar y 'q' para salir.")
    face_captured = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("No se pudo acceder a la cámara.")
            break
        cv2.imshow("Captura de Rostro", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('rostro_guardado.jpg', frame)
            st.write("Rostro guardado correctamente.")
            face_captured = True
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            st.write("Captura cancelada.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return face_captured

# Función para validar el rostro
def validate_face():
    if not os.path.exists('rostro_guardado.jpg'):
        st.write("No hay un rostro guardado para validar.")
        return False
    
    cap = cv2.VideoCapture(0)
    st.write("Capturando rostro para validación... Presiona 'q' para capturar.")
    valid_face = False

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("No se pudo acceder a la cámara.")
            break
        cv2.imshow("Validación de Rostro", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('rostro_validacion.jpg', frame)
            st.write("Rostro capturado para validación.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Cargar las imágenes
    rostro_guardado = face_recognition.load_image_file('rostro_guardado.jpg')
    rostro_validacion = face_recognition.load_image_file('rostro_validacion.jpg')

    # Obtener las codificaciones de los rostros
    try:
        encodings_guardado = face_recognition.face_encodings(rostro_guardado)[0]
        encodings_validacion = face_recognition.face_encodings(rostro_validacion)[0]
    except IndexError:
        st.write("No se detectó un rostro en una de las imágenes.")
        return False

    # Comparar los rostros
    results = face_recognition.compare_faces([encodings_guardado], encodings_validacion)

    if results[0]:
        st.write("Validación exitosa. Rostro reconocido.")
        valid_face = True
    else:
        st.write("Validación fallida. Rostro no reconocido.")
    
    return valid_face

# Interfaz de Streamlit
st.title("Sistema de Captura y Validación de Rostros")

option = st.selectbox("Seleccione una opción", ["Capturar Rostro", "Validar Rostro"])

if option == "Capturar Rostro":
    if capture_face():
        st.write("Proceso de captura completado.")
    else:
        st.write("Proceso de captura fallido o cancelado.")
elif option == "Validar Rostro":
    if validate_face():
        st.write("Acceso concedido.")
    else:
        st.write("Acceso denegado.")

