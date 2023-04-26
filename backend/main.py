from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173",
    "http://localhost:5500",
    "localhost:5500",
]

app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


@app.get('/')
async def main():
    return {'message': 'Working'}


@app.post("/solar-power-prediction/")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        df1 = pd.read_csv(file1.file)
        df2 = pd.read_csv(file2.file)
        frame = [df1, df2]
        df = pd.concat(frame)
        # # print(df.info())
        df = df.replace(np.nan, df.mean())
        # train = df.drop(['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY'], axis=1)
        # test = df['TOTAL_YIELD']
        # from sklearn.linear_model import LinearRegression
        # from sklearn.model_selection import train_test_split
        # le = LinearRegression()
        # x_train, x_test, y_train, y_test = train_test_split(
        #     train, test, test_size=0.2, random_state=100)
        # le.fit(x_train, y_train)
        # y_pred: int = le.predict(x_test)
        # # print(y_pred)
        # acc = le.score(x_test, y_test)
        # Return response
        return  {"df": df.to_json(orient="records")}
 
    except Exception as e:
        return {'error': e}