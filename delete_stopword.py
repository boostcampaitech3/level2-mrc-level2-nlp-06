"""
    passage의 불용어를 제거하는 기능을 수행합니다. (query에는 적용하지 않음)

        input: passage
        output: stopword가 제거된 passage 반환

"""
from konlpy.tag import Okt # pip install tweepy==3.10.0 호환성 문제 있다면 --> apt install default-jdk
from transformers import AutoTokenizer
from datasets import (
    Dataset,
    DatasetDict,
    load_from_disk
)
from tqdm import tqdm

class DeleteStopword:
    def __init__(self, tokenizer, stopwords_path:str = './stopwords.txt') -> None:
        self.stopwords = [] # text 파일 기반의 불용어 리스트입니다
        self.ad_stopwords = [] # Okt 형태소 분석 기반의 불용어(형용사, 부사) 리스트입니다.
        self.okt = Okt()
        self.tokenizer = tokenizer

        print("[INIT] DeleteStopword...")

        # stopword.txt 파일을 참조하여 불용어 리스트를 만듭니다.
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            temps = f.readlines()
            for word in temps:
                new_word = word.strip()
                self.stopwords.append(tokenizer.encode(new_word)[1]) # 불용어 토큰 리스트를 생성합니다.
        print("Done!")

    def remove_stopwords(self, passage:str) -> str:
        """
            입력으로 들어온 스트링에서 txt파일 기반 불용어 리스트(self.stopwords)에 해당하는 토큰을 제거한 후 반환합니다.
            input: str
            output: str - 불용어가 제거된 스트링
        """
        # passage에서 불필요한 심볼 '\n'을 제거합니다.
        passage = passage.replace('\\n', '')
        
        word_tokens = self.tokenizer.encode(passage)
        result = [word for word in word_tokens if not word in self.stopwords]

        return self.tokenizer.decode(result[1:-1])

    def konlpy_okt(self, passage:str) -> str:
        """
            Okt (형태소 분석기)를 이용해 passage에서 불필요한 부사, 형용사를 제거합니다.
        """
        # passage에서 불필요한 심볼 '\n'을 제거합니다.
        passage = passage.replace('\\n', '')
        okted_passage = self.okt.pos(passage)

        # 부사와 형용사 추출
        for word, pos in okted_passage:
            if pos == 'Adverb' or pos == 'Adjective':
                self.ad_stopwords.append(tokenizer.encode(word)[1])

        # 부사 및 형용사 제거
        word_tokens = self.tokenizer.encode(passage)
        #print(f'[raw text]: {passage}')
        #print(f'[tokens]: {word_tokens}')
        
        result = [word for word in word_tokens if not word in self.ad_stopwords]
        return self.tokenizer.decode(result[1:-1])

    def find_answer_index(self, passage:str, answer:str) -> int:
        """
            passage에서 answer word의 시작 인덱스를 찾아 반환합니다. 불용어 제거로 인해 answer text의 위치를 정정하는 작업입니다.
            이를 통해 새로운 데이터셋을 구축할 수 있습니다.
            return
                'answer_start' : passage에서 answer word가 시작되는 index입니다.
        """
        answer_start = passage.find(answer)
        if answer_start == -1:
            # answer를 못 찾는 경우가 있음 -> 불용어로 판단되서 제거되었기 때문(?) 혹은 answer가 너무 길어서...
            # 그냥 무시하기 or answer는 고정시키고 그 외의 불용어 제거하기
            pass

        return answer_start

    def refactt(self, example:Dataset)-> Dataset:
        """
            Dataset 클래스를 입력으로 받아 context의 불용어를 제거하고 asnwer_start를 조정하여 새로운 Dataset으로 반환합니다. 
            그 외의 항목들은 그대로 사용합니다.
        """
        example['context'] = self.konlpy_okt(example['context'])
        example['answers']['answer_start'][0] = self.find_answer_index(example['context'], example['answers']['text'][0])
        example['id'] = example['id'] + "-aug"

        return example
        

            


############
### DEMO ###
############
"""
    `python delete_stopword.py`로 실행하면 ./concat_datset 디렉토리에 [기존 데이터셋 + 불용어 제거된 데이터셋]이 등장합니다.
    ./new_dataset 디렉토리에는 [불용어 제거된 데이터셋] 만 저장됩니다.
"""


# 토크나이저 및 DeleteStopword class 호출
tokenizer = AutoTokenizer.from_pretrained(
    "klue/roberta-large",
    use_fast=True,
    )
ds = DeleteStopword(tokenizer=tokenizer)
dataset = load_from_disk("../data/train_dataset/") # 기존 데이터셋


if load_from_disk('./new_dataset') is None:
    new_dataset = dataset.map(ds.refactt) # 불용어 제거 시작
    new_dataset.save_to_disk('./new_dataset') # 불용어가 제거된 데이터셋을 디렉토리에 저장합니다.
else:
    new_dataset = load_from_disk('./new_dataset') # 불용어가 제거된 데이터셋이 이미 있다면 불러옵니다

print(dataset)
print(new_dataset)

err_count = 0

print('Working for Train set')
for new_data in tqdm(new_dataset['train']):
    # answer_start != -1 인 경우만 add item하기
    if new_data['answers']['answer_start'][0] != -1:
        dataset['train'] = dataset['train'].add_item(new_data) # datasets==1.7.v 부터 사용 가능합니다
    else:
        #print(f"{new_data['id']}는 답이 context에 존재하지 않음!")
        err_count += 1

print(f'[no-answer]: {err_count} 건에 대해서는 answer가 제거되어 적용하지 않았습니다.')
err_count = 0

print('Working for Validation set')
for new_data in tqdm(new_dataset['validation']):
    # answer_start != -1 인 경우만 add item하기
    if new_data['answers']['answer_start'][0] != -1:
        dataset['validation'] = dataset['validation'].add_item(new_data)
    else:
        #print(f"{new_data['id']}는 답이 context에 존재하지 않음!")
        err_count += 1
print(f'[no-answer]: {err_count} 건에 대해서는 answer가 제거되어 적용하지 않았습니다.')

#new_item = new_dataset['train'][0]
#dataset['train'] = dataset['train'].add_item(new_item) # datasets==1.7.부터 사용 가능
print('==========RESULT==============')
print(dataset)

dataset.save_to_disk('./concat_dataset')
print('save done...')





#print('================================================================')
#print(dataset['validation'][1])
#print('================================================================')
#print(new_dataset['validation'][1])
#print('================================================================')

#print(dataset['train'].keys())
#print(dataset['validation'].keys())
#print(new_dataset.keys())



#dataset.update(new_dataset)



#print(dataset['validation'][1])






"""
def add_prefix(example):
    example["text"] = "Review: " + example["text"]
    return example

dataset = dataset.map(add_prefix)
dataset["train"][0:3]["text"]

dataset = dataset.map(lambda example: tokenizer(example["text"]), batched=True)
dataset = dataset.map(add_prefix, num_proc=4)
"""