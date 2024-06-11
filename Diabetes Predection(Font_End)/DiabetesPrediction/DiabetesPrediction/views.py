from django.shortcuts import render


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(r"D:\Nsu\Spring 2024\CSE299\Datasets\Finalise DataFrame\diabetes_prediction_dataset.csv")

    df = df.drop_duplicates()

    df = df[df['smoking_history'] != "No Info"]
    df = df.replace({
        "Female": '0',
        "Male": '1',
        "Other": '2'
    })

    X = df.drop(columns=['diabetes', 'smoking_history'], axis=1)
    Y = df['diabetes']

    # stn = StandardScaler()
    # stn.fit(X)
    # standard_dt = stn.transform(X)
    # X = standard_dt

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    RFModel = RandomForestClassifier(n_estimators=40, criterion='entropy')
    RFModel.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])

    # user_input5 = (val1, val2, val3, val4, val5, val6, val7)
    # user_modify6 = np.asarray(user_input5)
    # user_reshape5 = user_modify6.reshape(1, -1)
    # std_user_data5 = stn.transform(user_reshape5)
    # pred = RFModel.predict(std_user_data5)
    pred = RFModel.predict([[val1, val2, val3, val4, val5, val6, val7]])
    result1 = ""
    if pred == [0]:
        result1 = " The patient has no diabetes"
    else:
        result1 = " The patient has diabetes"
    return render(request, "predict.html", {"result2": result1})
