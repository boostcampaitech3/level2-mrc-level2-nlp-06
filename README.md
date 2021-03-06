# ๐ก Machine Reading Comprehension (MRC) Project ๐ก


### ๐ฉ ํ๋ก์ ํธ ํ ๊ตฌ์ฑ ๋ฐ ์ญํ  ๐จ

|๊น์ ์ฌ|์ฐจ๊ฒฝ๋ฏผ|์ด๋ํ|๊ฐ์งํฌ|๊นํํ|
|--|--|--|--|--|
|Optimization, Model, Elastic Search:|Augementation, EDA, Model, Dense Retriever, Preprocessing|Optimization, EDA, BM25, Ensemble, Dense Retival|Optimization, Model, EDA, Sparse Retrival|EDA, Management, Question Generation| 
<br/>


### โ ํ๋ก์ ํธ ๊ฐ์

![image](https://user-images.githubusercontent.com/86389775/169706163-62a41196-9b2b-470f-89c5-caff728decc4.png)
<br/>
- ์ง๋ฌธ์ด ๋ฐ๋ก ์ฃผ์ด์ง์ง ์๊ณ  ์ฌ์ ์ ๊ตฌ์ถ๋์ด ์๋ Knowledge resource์์ ์ง๋ฌธ์ ๋ํด ๋๋ตํ  ์ ์๋ ๋ฌธ์๋ฅผ ์ฐพ์ ํ, ํด๋น ๋ฌธ์์์ ์ง๋ฌธ์ ๋ํ ๋๋ต์ ์ฐพ๋ ๋ฌธ์ 
- Query ๋ฌธ์ฅ ์๋ ฅ์ผ๋ก 2๋จ๊ณ Task ์ํ
    1. __Documnet Retriever__ - ๋ฌธ์ ์ฌ์ ์์ Query์ ๋๋ต ํ  ์ ์๋ ๋ฌธ์๋ฅผ ๊ฐ์ ธ์ค๋ ๋ชจ๋ธ
    2. __Document Reader__ - Retriever๋ก ๋ฐํ๋ ๋ฌธ์์์ Query์ ๊ฐ์ฅ ๊ด๋ จ ๊น์ Phase ์ฐพ๋ ๋ชจ๋ธ
<br/>    


### โ ํ๋ก์ ํธ ํ์๋ผ์ธ

|์ฃผ์ฐจ|์ํ ํญ๋ชฉ|
|----|----|
|1์ฃผ์ฐจ|์์จ์ ์ธ ์คํ ์์ด๋์ด ๊ณต์  ๋ฐ ์คํ ๊ณํ๊ณผ ์ํ, ๋ฒ ์ด์ค๋ผ์ธ ํ์ต ๋ฐ ๋ฐ์ดํฐ ์ฆ๊ฐ, ์คํ ํ๊ฒฝ ์ธํ|
|2์ฃผ์ฐจ|๋ชจ๋ธ ์ํคํ์ฒ ์ ํ ๋ฐ ๊ฒฐ๊ณผ ๊ณต์ , ๋ชจ๋ธ ์ต์ ํ, Retriever, Preprocessing, Elastic search ์คํ|
|3์ฃผ์ฐจ|Retriever, Preprocessing, Elastic search ์คํ, ๋ชจ๋ธ ์ต์ ํ ๋ฐ ์์๋ธ ์ํ|
<br/>


### โ ํ๋ก์ ํธ ์ํ ๊ฒฐ๊ณผ

๐ __Data Augementation__

1. ์ฃผ์ด์ง Question, Context ๋ฐ์ดํฐ ํฌ๊ธฐ๊ฐ ๋ค๋ฅธ pre-trained ๋ชจ๋ธ์ ์ฌ์ฉ๋ ๋ฐ์ดํฐ ์์ ๋นํด ๊ท๋ชจ๊ฐ ์๋ค ํ๋จ
2. ๊ธฐ์กด์ ๋ฐ์ดํฐ์ + KorQuAD ๋ฐ์ดํฐ์์ผ๋ก ํ์ต ์งํ์์ผฐ์ง๋ง ํจ๊ณผ ๋ฏธ๋ฏธ
3. Back-translation์ ํตํด ๋ฐ์ดํฐ ์ฆ๊ฐ ์๋ํ์ผ๋ ์ฑ๋ฅ ์ ํ ๊ธฐ๋ก<br/>

๐ __Preprocessing__

1. Wikipedia ๋ฐ์ดํฐ ์์ ํน์ ๋ฌธ์ ํฌํจ => Retrieval ์ฑ๋ฅ ์ ํ์ ์ํฅ ์ค ๊ฒ์ด๋ผ ํ๋จ
2. โ\nโ, โ#โ, ํ์ ๋ฑ ๋ฌธ์ ์ ๊ฑฐ
3. ๋์ผ ์กฐ๊ฑด ๋๋น 1 ~ 2% ์ฑ๋ฅ ํฅ์ ๊ธฐ๋ก<br/>

๐ __BM25__

![image](https://user-images.githubusercontent.com/86389775/169706099-9f6e5f0d-c2ec-4ff4-9709-cb4cef7728df.png)
<br/>  
1. ์ฃผ์ด์ง ์ฟผ๋ฆฌ์ ๋ํด ๋ฌธ์์ ์ฐ๊ด์ฑ์ ํ๊ฐํ๋ ๋ญํน ํจ์
2. TF-IDF์ ๊ฐ๋์ ๋ฐํ์ผ๋ก, ๋ฌธ์์ ๊ธธ์ด๊น์ง ๊ณ ๋ คํ์ฌ scoring ํ๋ฉฐ, TF ๊ฐ์ ํ๊ณ๋ฅผ ์ง์ ํด๋์ด ์ผ์ ํ ๋ฒ์๋ฅผ ์ ์ง
3. rank_bm25๋ฅผ ์ฌ์ฉํ์ฌ ๊ตฌํ, ๊ธฐ์กด ๋ชจ๋ธ ๋๋น EM score 15์  ๊ฐ๋ ์ฑ๋ฅ ํฅ์ ๊ธฐ๋ก<br/>

๐ __Dense Retrieval__

1. Passage์ Question์ ๊ฐ๊ฐ์ pretrained model์ ๋ง๋ค์ด์ hidden representation ๋ฒกํฐ๋ค์ ๋ด์ ์ ํตํด ๋ต์ ์ป์ ์ ์๋ passage์ question์ ๋ด์  ๊ฐ์ ๋์ด๋ ๋ฐฉํฅ์ผ๋ก ํ์ต ์ํค๋ ๋ฐฉ๋ฒ
2. [๋ผ๋ฌธ](https://arxiv.org/abs/2004.04906)์์ ์ ์ํ in-batch negative sampling ๊ธฐ๋ฒ์ ํตํด Dense Retrieval์ ๊ตฌํ ์๋  => Dataset size์ ํ๊ณ์ GPU resource์ ์ ํ์ผ๋ก ๋ผ๋ฌธ์์ ์ ์ํ batch size๋ก๋ ์ฌํ ์คํจ

<br/>

Public Score - EM : 59.1700, F1 : 69.0300 



![ํ๋ฉด ์บก์ฒ 2022-05-23 020842](https://user-images.githubusercontent.com/86389775/169707173-16d2c3dc-bf6f-4115-9f5d-034c55195d33.png)


Private Score - EM : 56.6700, F1 : 68.2300 


![ํ๋ฉด ์บก์ฒ 2022-05-23 020913](https://user-images.githubusercontent.com/86389775/169707195-ad29125d-d939-4d45-ad50-295bb44e86ac.png)



## ์ฝ๋ ์ค๋ช

### โ ์๊ตฌ ์ฌํญ

```
# data (51.2 MB)
tar -xzf data.tar.gz

# ํ์ํ ํ์ด์ฌ ํจํค์ง ์ค์น. 
bash ./install/install_requirements.sh
```

### โ ์ ์ฅ์ ๊ตฌ์กฐ

```bash
./assets/                # readme ์ ํ์ํ ์ด๋ฏธ์ง ์ ์ฅ
./install/               # ์๊ตฌ์ฌํญ ์ค์น ํ์ผ 
./data/                  # ์ ์ฒด ๋ฐ์ดํฐ. ์๋ ์์ธ ์ค๋ช
retrieval.py             # sparse retreiver ๋ชจ๋ ์ ๊ณต 
arguments.py             # ์คํ๋๋ ๋ชจ๋  argument๊ฐ dataclass ์ ํํ๋ก ์ ์ฅ๋์ด์์
trainer_qa.py            # MRC ๋ชจ๋ธ ํ์ต์ ํ์ํ trainer ์ ๊ณต.
utils_qa.py              # ๊ธฐํ ์ ํธ ํจ์ ์ ๊ณต 

train.py                 # MRC, Retrieval ๋ชจ๋ธ ํ์ต ๋ฐ ํ๊ฐ 
inference.py		     # ODQA ๋ชจ๋ธ ํ๊ฐ ๋๋ ์ ์ถ ํ์ผ (predictions.json) ์์ฑ
```

### โ ๋ฐ์ดํฐ ์๊ฐ

์๋๋ ์ ๊ณตํ๋ ๋ฐ์ดํฐ์์ ๋ถํฌ๋ฅผ ๋ณด์ฌ์ค๋๋ค.

![KakaoTalk_20220526_163504621](https://user-images.githubusercontent.com/86389775/170440940-16401b87-e0c6-4005-9bfd-d09b4551afe9.png)

๋ฐ์ดํฐ์์ ํธ์์ฑ์ ์ํด Huggingface ์์ ์ ๊ณตํ๋ datasets๋ฅผ ์ด์ฉํ์ฌ pyarrow ํ์์ ๋ฐ์ดํฐ๋ก ์ ์ฅ๋์ด์์ต๋๋ค. ๋ค์์ ๋ฐ์ดํฐ์์ ๊ตฌ์ฑ์๋๋ค.

```bash
./data/                        # ์ ์ฒด ๋ฐ์ดํฐ
    ./train_dataset/           # ํ์ต์ ์ฌ์ฉํ  ๋ฐ์ดํฐ์. train ๊ณผ validation ์ผ๋ก ๊ตฌ์ฑ 
    ./test_dataset/            # ์ ์ถ์ ์ฌ์ฉ๋  ๋ฐ์ดํฐ์. validation ์ผ๋ก ๊ตฌ์ฑ 
    ./wikipedia_documents.json # ์ํคํผ๋์ ๋ฌธ์ ์งํฉ. retrieval์ ์ํด ์ฐ์ด๋ corpus.
```

data์ ๋ํ argument ๋ `arguments.py` ์ `DataTrainingArguments` ์์ ํ์ธ ๊ฐ๋ฅํฉ๋๋ค. 


### โ Train

๋ง์ฝ arguments ์ ๋ํ ์ธํ์ ์ง์ ํ๊ณ  ์ถ๋ค๋ฉด `arguments.py` ๋ฅผ ์ฐธ๊ณ ํด์ฃผ์ธ์. 

roberta ๋ชจ๋ธ์ ์ฌ์ฉํ  ๊ฒฝ์ฐ tokenizer ์ฌ์ฉ์ ์๋ ํจ์์ ์ต์์ ์์ ํด์ผํฉ๋๋ค.
๋ฒ ์ด์ค๋ผ์ธ์ klue/bert-base๋ก ์งํ๋๋ ์ด ๋ถ๋ถ์ ์ฃผ์์ ํด์ ํ์ฌ ์ฌ์ฉํด์ฃผ์ธ์ ! 
tokenizer๋ train, validation (train.py), test(inference.py) ์ ์ฒ๋ฆฌ๋ฅผ ์ํด ํธ์ถ๋์ด ์ฌ์ฉ๋ฉ๋๋ค.
(tokenizer์ return_token_type_ids=False๋ก ์ค์ ํด์ฃผ์ด์ผ ํจ)

```python
# train.py
def prepare_train_features(examples):
        # truncation๊ณผ padding(length๊ฐ ์งง์๋๋ง)์ ํตํด toknization์ ์งํํ๋ฉฐ, stride๋ฅผ ์ด์ฉํ์ฌ overflow๋ฅผ ์ ์งํฉ๋๋ค.
        # ๊ฐ example๋ค์ ์ด์ ์ context์ ์กฐ๊ธ์ฉ ๊ฒน์น๊ฒ๋ฉ๋๋ค.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta๋ชจ๋ธ์ ์ฌ์ฉํ  ๊ฒฝ์ฐ False, bert๋ฅผ ์ฌ์ฉํ  ๊ฒฝ์ฐ True๋ก ํ๊ธฐํด์ผํฉ๋๋ค.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

```bash
# ํ์ต ์์ (train_dataset ์ฌ์ฉ)
python train.py --output_dir ./models/train_dataset --num_train_epochs 10 --do_train
```

```bash
# Retriever ํ๋ จ(?)
python train.py --train_retrieval
```

### โ Eval

MRC ๋ชจ๋ธ์ ํ๊ฐ๋(`--do_eval`) ๋ฐ๋ก ์ค์ ํด์ผ ํฉ๋๋ค.  ์ ํ์ต ์์์ ๋จ์ํ `--do_eval` ์ ์ถ๊ฐ๋ก ์๋ ฅํด์ ํ๋ จ ๋ฐ ํ๊ฐ๋ฅผ ๋์์ ์งํํ  ์๋ ์์ต๋๋ค.

```bash
# mrc ๋ชจ๋ธ ํ๊ฐ (train_dataset ์ฌ์ฉ)
python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### โ Inference

retrieval ๊ณผ mrc ๋ชจ๋ธ์ ํ์ต์ด ์๋ฃ๋๋ฉด `inference.py` ๋ฅผ ์ด์ฉํด odqa ๋ฅผ ์งํํ  ์ ์์ต๋๋ค.

* ํ์ตํ ๋ชจ๋ธ์  test_dataset์ ๋ํ ๊ฒฐ๊ณผ๋ฅผ ์ ์ถํ๊ธฐ ์ํด์  ์ถ๋ก (`--do_predict`)๋ง ์งํํ๋ฉด ๋ฉ๋๋ค. 

* ํ์ตํ ๋ชจ๋ธ์ด train_dataset ๋ํด์ ODQA ์ฑ๋ฅ์ด ์ด๋ป๊ฒ ๋์ค๋์ง ์๊ณ  ์ถ๋ค๋ฉด ํ๊ฐ(`--do_eval`)๋ฅผ ์งํํ๋ฉด ๋ฉ๋๋ค.

* inference ๊ฒฐ๊ณผ๋ฌผ json ํ์ผ์ prefix๋ฅผ ๋ถ์ผ ์ ์์ต๋๋ค! (`--name`)์ ์ํ๋ ๋ช์นญ์ ์ ์ผ์๋ฉด predictions_{name}.json ํํ๋ก ์ ์ฅ๋ฉ๋๋ค. ์ถํ ensemble์ ์ํด ์ ๋ณด๊ดํด์ฃผ์ธ์!

```bash
# ODQA ์คํ (test_dataset ์ฌ์ฉ)
# wandb ๊ฐ ๋ก๊ทธ์ธ ๋์ด์๋ค๋ฉด ์๋์ผ๋ก ๊ฒฐ๊ณผ๊ฐ wandb ์ ์ ์ฅ๋ฉ๋๋ค. ์๋๋ฉด ๋จ์ํ ์ถ๋ ฅ๋ฉ๋๋ค
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict --overwrite_output_dir --name new_case
```

```bash
# ํ๊ฐ(--do_eval)์ ์งํํ๊ณ  ์ถ์ ๋๋ train dataset์ผ๋ก...
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/train_dataset --model_name_or_path ./models/train_dataset --do_eval --overwrite_output_dir --name new_case
`train_dataset์ ์ฌ์ฉํจ!!`
```

### โ How to submit

`inference.py` ํ์ผ์ ์ ์์์ฒ๋ผ `--do_predict` ์ผ๋ก ์คํํ๋ฉด `--output_dir` ์์น์ `predictions.json` ์ด๋ผ๋ ํ์ผ์ด ์์ฑ๋ฉ๋๋ค. ํด๋น ํ์ผ์ ์ ์ถํด์ฃผ์๋ฉด ๋ฉ๋๋ค.

### โ Things to know

1. `train.py` ์์ sparse embedding ์ ํ๋ จํ๊ณ  ์ ์ฅํ๋ ๊ณผ์ ์ ์๊ฐ์ด ์ค๋ ๊ฑธ๋ฆฌ์ง ์์ ๋ฐ๋ก argument ์ default ๊ฐ True๋ก ์ค์ ๋์ด ์์ต๋๋ค. ์คํ ํ sparse_embedding.bin ๊ณผ tfidfv.bin ์ด ์ ์ฅ์ด ๋ฉ๋๋ค. **๋ง์ฝ sparse retrieval ๊ด๋ จ ์ฝ๋๋ฅผ ์์ ํ๋ค๋ฉด, ๊ผญ ๋ ํ์ผ์ ์ง์ฐ๊ณ  ๋ค์ ์คํํด์ฃผ์ธ์!** ์๊ทธ๋ฌ๋ฉด ๊ธฐ์กด ํ์ผ์ด load ๋ฉ๋๋ค.

2. ๋ชจ๋ธ์ ๊ฒฝ์ฐ `--overwrite_cache` ๋ฅผ ์ถ๊ฐํ์ง ์์ผ๋ฉด ๊ฐ์ ํด๋์ ์ ์ฅ๋์ง ์์ต๋๋ค. 

3. `./outputs/` ํด๋ ๋ํ `--overwrite_output_dir` ์ ์ถ๊ฐํ์ง ์์ผ๋ฉด ๊ฐ์ ํด๋์ ์ ์ฅ๋์ง ์์ต๋๋ค.
