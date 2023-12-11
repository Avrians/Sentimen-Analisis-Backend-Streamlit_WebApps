# Pembuatan Dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#Implementasi Model
import nltk
import re
import pickle
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer

#DBMS
import pymysql

def is_table_empty(table, host='localhost', user='root', password='', database='review'):
    # Establish a connection to the MySQL database
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    
    # Create a cursor object to execute SQL queries
    cursor = connection.cursor()
    
    # Check if the table is empty
    query_check_empty = f"SELECT COUNT(*) FROM {table}"
    cursor.execute(query_check_empty)
    count_result = cursor.fetchone()[0]

    # Close the cursor and the database connection
    cursor.close()
    connection.close()

    return count_result == 0

#Implemnetasi Model dengan Data Baru
def read_mysql_table(table, host='localhost', user='root', password='', database='review'):
    # Establish a connection to the MySQL database
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    
    # Create a cursor object to execute SQL queries
    cursor = connection.cursor()
    
    query = f"SELECT * FROM {table}"
    cursor.execute(query)
    result = cursor.fetchall()
    
    # Convert the result to a Pandas DataFrame
    df = pd.DataFrame(result)
    
    # Assign column names based on the cursor description
    df.columns = [column[0] for column in cursor.description]
    
    # Close the cursor and the database connection
    cursor.close()
    connection.close()
    
    return df

table_name = 'input_review'

if not is_table_empty(table_name):
    df = read_mysql_table(table_name)
    # #menyimpan tweet. (tipe data series pandas)
    

    # Specify the file path of the pickle file
    file_path = 'Model/training.pickle'

    # Read the pickle file
    with open(file_path, 'rb') as file:
        data_train = pickle.load(file)
        
    # pembuatan vector kata
    vectorizer = TfidfVectorizer()
    train_vector = vectorizer.fit_transform(data_train)
    reviews2 = [" ".join(r) for r in reviews]

    load_model = pickle.load(open('Model/tfidf_Model Naive Bayes_nvb.pkl','rb'))

    result = []

    for test in reviews2:
        test_data = [str(test)]
        test_vector = vectorizer.transform(test_data)
        pred = load_model.predict(test_vector)
        result.append(pred[0])
        
    unique_labels(result)

    df['label'] = result

    def delete_all_data_from_table(table, host='localhost', user='root', password='', database='review'):
        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()
        
        # Delete all data from the specified table
        query = f"DELETE FROM {table}"
        cursor.execute(query)
        
        # Commit the changes
        connection.commit()
        
        # Close the cursor and the database connection
        cursor.close()
        connection.close()

    delete_all_data_from_table('input_review')

    def insert_df_into_hasil_model(df, host='localhost', user='root', password='', database='review'):
        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        # Insert each row from the DataFrame into the 'hasil_model' table
        for index, row in df.iterrows():
            query = "INSERT INTO hasil_model (id_review, nama, tanggal, review, label) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(query, (row['id_review'], row['nama'], row['tanggal'], row['review'], row['label']))

        # Commit the changes
        connection.commit()

        # Close the cursor and the database connection
        cursor.close()
        connection.close()

    insert_df_into_hasil_model(df)

    table_name = 'hasil_model'
    hasil_df = read_mysql_table(table_name)
    hasil_df.to_csv('Data/hasil_model.csv')
    data = pd.read_csv('data/hasil_model.csv')
else:
    # Membaca data dari file CSV
    data = pd.read_csv('data/hasil_model.csv')

data = data[['review', 'label']]

# Menghitung jumlah data dengan label positif, negatif, dan netral
jumlah_positif = len(data[data['label'] == 1])
jumlah_negatif = len(data[data['label'] == 0])
jumlah_netral = len(data[data['label'] == -1])

# Membuat dashboard Streamlit
st.title('Dashboard Analisis Sentimen Naive Bayes')

# Membuat kolom-kolom untuk menyusun metrik
col1, col2, col3 = st.columns(3)

# Menampilkan metrik untuk jumlah data dengan label positif, negatif, dan netral
with col1:
    st.metric(label='Positif (1)', value=jumlah_positif)

with col2:
    st.metric(label='Negatif (0)', value=jumlah_negatif)

with col3:
    st.metric(label='Netral (-1)', value=jumlah_netral)

# Membuat bar chart dengan warna yang berbeda
fig, ax = plt.subplots()
labels = ['Positif (1)', 'Negatif (0)', 'Netral (-1)']
jumlah_data = [jumlah_positif, jumlah_negatif, jumlah_netral]
colors = ['green', 'red', 'gray']
ax.bar(labels, jumlah_data, color=colors)
st.pyplot(fig)