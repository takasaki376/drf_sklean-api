import os
import pickle
import pandas as pd
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from collections import defaultdict

class IrisPredict(views.APIView):

    def __init__(self):
        super(IrisPredict, self).__init__()
        self.model_name = "iris_model"

    # =====================================
    # 予測する
    # =====================================
    def post(self, request):

        # フロント側へ返す値の格納領域
        predictions = defaultdict(int)
        se_setosa = defaultdict(int)
        se_versicolor = defaultdict(int)
        se_virginica = defaultdict(int)
        pe_setosa = defaultdict(int)
        pe_versicolor = defaultdict(int)
        pe_virginica = defaultdict(int)
        initData = defaultdict(int)
        initData['x'] = None
        initData['y'] = None
        predData = defaultdict(int)

        # 学習済のモデル読み込み
        path = os.path.join(settings.MODEL_ROOT, self.model_name)
        with open(path, 'rb') as file:
            model = pickle.load(file)

        # 予測用に項目名変更（フロント側で項目名にカッコが使えなかったため、予測データに合わせる）
        x_pred = pd.DataFrame(request.data)
        x_pred = x_pred.rename(columns={'sepal_length':'sepal length (cm)','sepal_width':'sepal width (cm)','petal_length':'petal length (cm)','petal_width':'petal width (cm)' })

        # 予測する
        try:
            result = model.predict(x_pred)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        # リクエストデータの編集（フロント側でChart.jsに編集可能な形に変換する）
        if result[0] == 0:
            se_setosa['x'] = x_pred.iat[0, 0]
            se_setosa['y'] = x_pred.iat[0, 1]
            pe_setosa['x'] = x_pred.iat[0, 2]
            pe_setosa['y'] = x_pred.iat[0, 3]
            se_versicolor = initData
            pe_versicolor = initData
            se_virginica = initData
            pe_virginica = initData
        if result[0] == 1:
            se_versicolor['x'] = x_pred.iat[0, 0]
            se_versicolor['y'] = x_pred.iat[0, 1]
            pe_versicolor['x'] = x_pred.iat[0, 2]
            pe_versicolor['y'] = x_pred.iat[0, 3]
            se_setosa = initData
            pe_setosa = initData
            se_virginica = initData
            pe_virginica = initData
        if result[0] == 2:
            se_virginica['x'] = x_pred.iat[0, 0]
            se_virginica['y'] = x_pred.iat[0, 1]
            pe_virginica['x'] = x_pred.iat[0, 2]
            pe_virginica['y'] = x_pred.iat[0, 3]
            se_setosa = initData
            pe_setosa = initData
            se_versicolor = initData
            pe_versicolor = initData
        predData['sepal_length'] = x_pred.iat[0, 0]
        predData['sepal_width'] = x_pred.iat[0, 1]
        predData['petal_length'] = x_pred.iat[0, 2]
        predData['petal_width'] = x_pred.iat[0, 3]
        predData['pred'] = result[0]
        predictions['se_setosa'] = se_setosa.copy()
        predictions['se_versicolor'] = se_versicolor.copy()
        predictions['se_virginica'] = se_virginica.copy()
        predictions['pe_setosa'] = pe_setosa.copy()
        predictions['pe_versicolor'] = pe_versicolor.copy()
        predictions['pe_virginica'] = pe_virginica.copy()
        predictions['pred'] = predData

        # 予測結果を返す
        return Response(predictions, status=status.HTTP_200_OK)
