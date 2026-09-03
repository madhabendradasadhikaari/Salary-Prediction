from flask import Flask,request,render_template
import pickle
import numpy as np

app=Flask(__name__)

with open('salary.pkl','rb') as model_files:
	ml_model=pickle.load(model_files)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/process",methods=['POST'])
def process():
	experience = float(request.form['Experience'])

	import_data=np.array([[experience]])

	result=ml_model.predict(import_data)[0]

	formatted_salary = f"${result:,.2f}"

	print("Predicted Salary:",formatted_salary)

	return render_template("myresult.html",prediction=formatted_salary)

if __name__=="__main__":
	app.run(debug=True)