import os  # para reconocer el entorno y usarlo como consola
from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
import csv
import joblib
import json

# unix/mac
# export FLASK_APP=myapp
# export FLASK_ENV="development" #entorno de desarrollo

# windows
# set FLASK_APP=myapp
# set FLASK_ENV=development #entorno de desarrollo
# flask run http://localhost:3000

app = Flask(__name__)



@app.route('/')
def index():
    return render_template("index.html")

model = None
model = joblib.load(open("model.pkl", "rb")) 

""" # reading the data in the csv file
df = pd.read_csv('iris.csv')
df.to_csv('iris.csv', index=None) """

@app.route('/preview')
def preview():
    # reading the data in the csv file
    df = pd.read_csv('iris.csv')
    df.to_csv('iris.csv', index=None)
    # Convert pandas dataframe to html table flask
    df_html = df.to_html()
    return render_template('preview.html', data=df_html)

@app.route('/',methods=["POST"])
def analyze():
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']
		#model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		result_prediction = model.predict(ex1)
  
		# Reloading the Model
		return render_template('index.html', 
                         		petal_width=petal_width, 
								sepal_width=sepal_width,
								sepal_length=sepal_length,
								petal_length=petal_length,
								clean_data=clean_data,
								result_prediction=result_prediction)
  

@app.route('/iris/', methods=['GET'])
def irisData():
    data = pd.read_csv('iris.csv')
    describe = data.describe().to_json(orient="index")
    describe = json.loads(describe)
    return describe


@app.route('/irisdata/', methods=['POST'])
def insertData():
    data = request.data
    data = json.loads(data)
    with open('iris.csv', 'a', newline='') as csvfile:
        fieldnames = ['sepal_length', 'sepal_width', 'petal_length',
                      'petal_width', 'species']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'sepal_length': data['sepal_length'],
                        'sepal_width': data['sepal_width'],
                         'petal_length': data['petal_length'],
                         'petal_width': data['petal_width'],
                         'species': data['species']})
        print("writing complete")
    return data

@app.route('/updateData/', methods=['PUT'])
def updatedata():
        # definir 'data' como el conjunto de datos
        # que se reciben a través del Postman:
    data = request.data
    data = json.loads(data)
    df = pd.read_csv('iris.csv')
        # sustituimos la última fila del dataset cada uno de los valores
        # con los datos que recibimos 'data':
    df.loc[df.index[-1], 'sepal_length'] = data['sepal_length']
    df.loc[df.index[-1], 'sepal_width'] = data['sepal_width']
    df.loc[df.index[-1], 'petal_length'] = data['petal_length']
    df.loc[df.index[-1], 'petal_width'] = data['petal_width']
    df.loc[df.index[-1], 'species'] = data['species']
        # convertir a csv
    df.to_csv('iris.csv', index=False)
        # mostrar el último dato en formato Json:
    result = df.iloc[-1].to_json(orient="index")
    return result

"""
@app.route('/deleteData/', methods=['DELETE'])
def deleteData(item_id: int):
     Método *DELETE* para eliminar 1 dato en el dataset según ID: 
        * item_ID requerido.
        
    df = pd.read_csv('iris.csv')  # Leemos el csv con ayuda de pandas:
    df.drop(df.index[item_id], inplace=True)  # Eliminar la última fila
    df.to_csv('iris.csv', index=False)  # convertir a csv
    return 'Eliminado'
"""


# Ruta de inicio "/deleteData/", metodo DELETE
@app.route('/deleteData/', methods=['DELETE'])
def deleteData():
    df = pd.read_csv('iris.csv')
    # Eliminar la última fila
    df.drop(df.index[-1], inplace=True)
    # convertir a csv
    df.to_csv('iris.csv', index=False)
    # mostrar el último dato en formato Json:
    result = df.iloc[-1].to_json(orient="index")
    result = json.loads(result)
    return "Eliminado"

if __name__ == '__main__':
    app.run(debug=True)
