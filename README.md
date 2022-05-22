# 💡 Machine Reading Comprehension (MRC) Project 💡


### 👩 프로젝트 팀 구성 및 역할 👨

|김선재|차경민|이도훈|강진희|김태훈|
|--|--|--|--|--|
|Optimization, Model, Elastic Search:|Augementation, EDA, Model, Dense Retriever, Preprocessing|Optimization, EDA, BM25, Ensemble, Dense Retival|Optimization, Model, EDA, Sparse Retrival|EDA, Management, Question Generation| 
<br/>


### ✅ 프로젝트 개요

![image](https://user-images.githubusercontent.com/86389775/169706163-62a41196-9b2b-470f-89c5-caff728decc4.png)
<br/>
- 지문이 따로 주어지지 않고 사전에 구축되어 있는 Knowledge resource에서 질문에 대해 대답할 수 있는 문서를 찾은 후, 해당 문서에서 질문에 대한 대답을 찾는 문제
- Query 문장 입력으로 2단계 Task 수행
    1. __Documnet Retriever__ - 문서 사전에서 Query에 대답 할 수 있는 문서를 가져오는 모델
    2. __Document Reader__ - Retriever로 반환된 문서에서 Query와 가장 관련 깊은 Phase 찾는 모델
<br/>    


### ✅ 프로젝트 타임라인

|주차|수행 항목|
|----|----|
|1주차|자율적인 실험 아이디어 공유 및 실험 계획과 수행, 베이스라인 학습 및 데이터 증강, 실험 환경 세팅|
|2주차|모델 아키텍처 선택 및 결과 공유, 모델 최적화, Retriever, Preprocessing, Elastic search 실험|
|3주차|Retriever, Preprocessing, Elastic search 실험, 모델 최적화 및 앙상블 수행|
<br/>


### ✅ 프로젝트 수행 결과

📌 __Data Augementation__

1. 주어진 Question, Context 데이터 크기가 다른 pre-trained 모델에 사용된 데이터 셋에 비해 규모가 작다 판단
2. 기존의 데이터셋 + KorQuAD 데이터셋으로 학습 진행시켰지만 효과 미미
3. Back-translation을 통해 데이터 증강 시도했으나 성능 저하 기록<br/>

📌 __Preprocessing__

1. Wikipedia 데이터 셋에 특수 문자 포함 => Retrieval 성능 저하에 영향 줄 것이라 판단
2. ‘\n’, ‘#’, 한자 등 문자 제거
3. 동일 조건 대비 1 ~ 2% 성능 향상 기록<br/>

📌 __BM25__

![image](https://user-images.githubusercontent.com/86389775/169706099-9f6e5f0d-c2ec-4ff4-9709-cb4cef7728df.png)
<br/>  
1. 주어진 쿼리에 대해 문서의 연관성을 평가하는 랭킹 함수
2. TF-IDF의 개념을 바탕으로, 문서의 길이까지 고려하여 scoring 하며, TF 값에 한계를 지정해두어 일정한 범위를 유지
3. rank_bm25를 사용하여 구현, 기존 모델 대비 EM score 15점 가량 성능 향상 기록<br/>

📌 __Dense Retrieval__

1. Passage와 Question을 각각의 pretrained model을 만들어서 hidden representation 벡터들의 내적을 통해 답을 얻을 수 있는 passage와 question의 내적 값을 높이는 방향으로 학습 시키는 방법
2. [논문](https://arxiv.org/abs/2004.04906)에서 제시한 in-batch negative sampling 기법을 통해 Dense Retrieval을 구현 시도  => Dataset size의 한계와 GPU resource의 제한으로 논문에서 제시한 batch size로는 재현 실패

<br/>



## 코드 설명

### ✅ 요구 사항

```
# data (51.2 MB)
tar -xzf data.tar.gz

# 필요한 파이썬 패키지 설치. 
bash ./install/install_requirements.sh
```

### ✅ 저장소 구조

```bash
./assets/                # readme 에 필요한 이미지 저장
./install/               # 요구사항 설치 파일 
./data/                  # 전체 데이터. 아래 상세 설명
retrieval.py             # sparse retreiver 모듈 제공 
arguments.py             # 실행되는 모든 argument가 dataclass 의 형태로 저장되어있음
trainer_qa.py            # MRC 모델 학습에 필요한 trainer 제공.
utils_qa.py              # 기타 유틸 함수 제공 

train.py                 # MRC, Retrieval 모델 학습 및 평가 
inference.py		     # ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성
```

### ✅ 데이터 소개

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![데이터 분포](./assets/dataset.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```bash
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성 
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 


### ✅ Train

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요. 

roberta 모델을 사용할 경우 tokenizer 사용시 아래 함수의 옵션을 수정해야합니다.
베이스라인은 klue/bert-base로 진행되니 이 부분의 주석을 해제하여 사용해주세요 ! 
tokenizer는 train, validation (train.py), test(inference.py) 전처리를 위해 호출되어 사용됩니다.
(tokenizer의 return_token_type_ids=False로 설정해주어야 함)

```python
# train.py
def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

```bash
# 학습 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --num_train_epochs 10 --do_train
```

```bash
# Retriever 훈련(?)
python train.py --train_retrieval
```

### ✅ Eval

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다.  위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```bash
# mrc 모델 평가 (train_dataset 사용)
python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### ✅ Inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

* 학습한 모델의  test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다. 

* 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`--do_eval`)를 진행하면 됩니다.

* inference 결과물 json 파일에 prefix를 붙일 수 있습니다! (`--name`)에 원하는 명칭을 적으시면 predictions_{name}.json 형태로 저장됩니다. 추후 ensemble을 위해 잘 보관해주세요!

```bash
# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict --overwrite_output_dir --name new_case
```

```bash
# 평가(--do_eval)을 진행하고 싶을 때는 train dataset으로...
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/train_dataset --model_name_or_path ./models/train_dataset --do_eval --overwrite_output_dir --name new_case
`train_dataset을 사용함!!`
```

### ✅ How to submit

`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--output_dir` 위치에 `predictions.json` 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

### ☑️ Things to know

1. `train.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 기존 파일이 load 됩니다.

2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 

3. `./outputs/` 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.
