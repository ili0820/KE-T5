# KE-T5
---
2021 국립국어원 인공 지능 언어 능력 평가 중 

판정의문문(boolq), 동형이의어(WIC)과제에 알맞게 finetune 해보았다.

pytorch_lightning을 기반으로 작성

hugging face 에서  model_name = 'KETI-AIR/ke-t5-base' 으로 불러와서 사용

Preprocess
---
preprocess.py를 사용하여 boolq 데이터를 전처리.

1. Answer의 값들을 1에서 "예"로, 0에서 "아니요"로 변경
2. Column 명을 "Answer(FALSE = 0, TRUE = 1)"에서 "answer_text"로 변경
3. 필요없는 ID column 제거
4. 추후 학습 파라미터로 사용할 max_len를 구하기 위하여 Text의 최대 길이 cnt_train으로 저장
5. val,test 데이터동일하게 처리

Main
---
1. 혹시몰라서 다시 text의 최대 값을 구하여 사용했다.(없어도 될듯)
2. BoolqDataset 정의 tokenizer,최대 길이를 넣어주고, source targget encoding을 진행하였다. source encoding은 질문과 문단을 넣어주었고 target encoding은 answer_test를 넣어주었다.
3. BoolqDataModule정의 각각의 dataloader 를 정의하고 데이터를 입력
4. BoolqModel정의 forward 정의,각각의 training,valdation,test step 정의, Optimizer AdamW사용
5. Freeze를 사용하여 학습(in progress)
6. main 정의 Batch size, n_epochs 설정 및 train.

Test
---
1. trained_model을 불러와서 test_data로 테스트.

jsontest
---
제출용 더미 json파일 생성

accuracy
--- 
devSet으로 간단하게 accuracy 측정

WIC&WIC_TEST
---
WIC과제를 위한 finetune (in progress)

참고자료
---
https://youtu.be/r6XY80Z9eSA
