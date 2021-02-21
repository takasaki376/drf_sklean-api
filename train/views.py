import os
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from sklearn import svm
from collections import defaultdict

class IrisTrain(views.APIView):

    def __init__(self):
        super(IrisTrain, self).__init__()
        self.model_name = "iris_model"

    # =====================================
    # 学習に使用したデータをフロントへ渡す
    # =====================================
    def get(self, request):
        # テストデータの読み込み
        iris = datasets.load_iris()

        # 入力データ
        X = pd.DataFrame(iris.data, columns=iris.feature_names)

        # 入力データと出力データを結合する。
        df_iris = X
        df_iris['target'] = iris.target

        # リクエストデータの編集用変数
        se_setosa_list = []
        se_versicolor_list = []
        se_virginica_list = []
        se_setosa = defaultdict(int)
        se_versicolor = defaultdict(int)
        se_virginica = defaultdict(int)
        pe_setosa_list = []
        pe_versicolor_list = []
        pe_virginica_list = []
        pe_setosa = defaultdict(int)
        pe_versicolor = defaultdict(int)
        pe_virginica = defaultdict(int)
        list = defaultdict(dict)

        # リクエストデータの編集（フロント側でChart.jsに編集可能な形に変換する）
        for data in df_iris.to_records():
            if data[5] == 0:
                se_setosa['x'] = data[1]
                se_setosa['y'] = data[2]
                se_setosa_list.append(se_setosa.copy())
                pe_setosa['x'] = data[3]
                pe_setosa['y'] = data[4]
                pe_setosa_list.append(pe_setosa.copy())
            if data[5] == 1:
                se_versicolor['x'] = data[1]
                se_versicolor['y'] = data[2]
                se_versicolor_list.append(se_versicolor.copy())
                pe_versicolor['x'] = data[3]
                pe_versicolor['y'] = data[4]
                pe_versicolor_list.append(pe_versicolor.copy())
            if data[5] == 2:
                se_virginica['x'] = data[1]
                se_virginica['y'] = data[2]
                se_virginica_list.append(se_virginica.copy())
                pe_virginica['x'] = data[3]
                pe_virginica['y'] = data[4]
                pe_virginica_list.append(pe_virginica.copy())

        list['se_setosa'] = se_setosa_list.copy()
        list['se_versicolor'] = se_versicolor_list.copy()
        list['se_virginica'] = se_virginica_list.copy()
        list['pe_setosa'] = pe_setosa_list.copy()
        list['pe_versicolor'] = pe_versicolor_list.copy()
        list['pe_virginica'] = pe_virginica_list.copy()

        # 学習データをフロントへ返す
        return Response(list, status=status.HTTP_200_OK)

    # =====================================
    # 学習する
    # =====================================
    def post(self, request):
        # テストデータの読み込み
        iris = datasets.load_iris()
        # 出力データを出力値と名称の辞書型に変換
        mapping = dict(zip(np.unique(iris.target), iris.target_names))
        print(mapping)

        # 入力データ
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        # 出力データ（正解）
        y = pd.DataFrame(iris.target)

        # 学習する
        try:
            # 学習するモデルの定義（今回はサポートベクターマシン）
            clf = svm.SVC()
            # 入力データに対して正解を読み込ませて学習する
            clf.fit(X, y)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        # 学習結果もモデルをファイル保存する
        path = os.path.join(settings.MODEL_ROOT, self.model_name)
        with open(path, 'wb') as file:
            pickle.dump(clf, file)

        # フロントエンドへステータスだけを返す
        return Response(status=status.HTTP_200_OK)