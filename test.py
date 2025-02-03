from flask import Flask

#Create a Flask app instance
app = Flask(__name__)

#Define the app if this file is executed
@app.route('/')
def hellow():
    return 'Hello, World!'

#Run the app if this file is executed   
if __name__ == '__main__':
    app.run(debug=True)
