from flask import Flask,request,render_template,jsonify
from main_folder.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        print(request.get_json())
        data = request.get_json()
        data=CustomData(
            Delivery_person_Age=float(data['Delivery_person_Age']),
            Delivery_person_Ratings = float(data['Delivery_person_Ratings']),
            Restaurant_latitude = float(data['Restaurant_latitude']),
            Restaurant_longitude = float(data['Restaurant_longitude']),
            Delivery_location_latitude = float(data['Delivery_location_latitude']),
            Delivery_location_longitude = float(data['Delivery_location_longitude']),
            Weather_conditions = data['Weather_conditions'],
            Road_traffic_density= data['Road_traffic_density'],
            Vehicle_condition = float(data['Vehicle_condition']),
            Type_of_order = data['Type_of_order'],
            Type_of_vehicle= data['Type_of_vehicle'],
            multiple_deliveries = float(data['multiple_deliveries']),
            Festival = data['Festival'],
            City= data['City'],
            Time_diff = float(data['Time_Diff'])
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=int(round(pred[0],2))
        
        print(results)
        
        response = {
            "result": results
        }

        return response


if __name__=="__main__":
    app.run(host='localhost',port=3000,debug=True)

