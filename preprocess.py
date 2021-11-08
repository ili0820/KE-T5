import pandas as pd

pd.set_option('mode.chained_assignment', None)
def preprocess():
    train_data = pd.read_csv("dataset/boolq/SKT_BoolQ_Train.tsv", sep="\t")
    for i in range(len(train_data["Answer(FALSE = 0, TRUE = 1)"])):
        if train_data["Answer(FALSE = 0, TRUE = 1)"][i] == 1:
            train_data["Answer(FALSE = 0, TRUE = 1)"][i] = "예"
        else:
            train_data["Answer(FALSE = 0, TRUE = 1)"][i] = "아니요"

    train_data = train_data.rename(columns={"Answer(FALSE = 0, TRUE = 1)": "answer_text"})
    train_data.drop(columns=["ID"], inplace=True)
    train_data = train_data.astype(str)

    cnt_train = 0
    for i in train_data["Text"]:
        if len(i) > cnt_train:
            cnt_train = len(i)

    cnt_train_A = 0
    for i in train_data["answer_text"]:
        if len(i) > cnt_train_A:
            cnt_train_A = len(i)


    val_data = pd.read_csv("dataset/boolq/SKT_BoolQ_Dev.tsv", sep="\t")
    for i in range(len(val_data["Answer(FALSE = 0, TRUE = 1)"])):
        if val_data["Answer(FALSE = 0, TRUE = 1)"][i] == 1:
            val_data["Answer(FALSE = 0, TRUE = 1)"][i] = "맞습니다"
        else:
            val_data["Answer(FALSE = 0, TRUE = 1)"][i] = "틀립니다"
    val_data = val_data.rename(columns={"Answer(FALSE = 0, TRUE = 1)": "answer_text"})
    val_data.drop(columns=["ID"], inplace=True)
    val_data = val_data.astype(str)

    cnt_val = 0
    for i in val_data["Text"]:
        if len(i) > cnt_val:
            cnt_val = len(i)

    cnt_val_A = 0
    for i in val_data["answer_text"]:
        if len(i) > cnt_val_A:
            cnt_val_A = len(i)



    test_data = pd.read_csv("dataset/boolq/SKT_BoolQ_Test.tsv", sep="\t")

    test_data = test_data.rename(columns={"Answer(FALSE = 0, TRUE = 1)": "answer_text"})
    test_data.drop(columns=["ID"], inplace=True)
    test_data = test_data.astype(str)
    cnt_test=0
    for i in test_data["Text"]:
        if len(i) > cnt_test:
            cnt_test = len(i)


    return pd.DataFrame(train_data),pd.DataFrame(val_data),pd.DataFrame(test_data),max(cnt_train,cnt_val,cnt_test),max(cnt_train_A,cnt_val_A)


