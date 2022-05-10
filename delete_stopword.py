"""
    passage의 불용어를 제거하는 기능을 수행합니다. (query에는 적용하지 않음)

        input: passage
        output: stopword가 제거된 passage 반환

"""
from konlpy.tag import Okt # pip install tweepy==3.10.0 호환성 문제 있음 apt install default-jdk
from transformers import AutoTokenizer

class DeleteStopword:
    def __init__(self, tokenizer, stopwords_path:str = './stopwords.txt') -> None:
        self.stopwords = []
        self.ad_stopwords = []
        self.okt = Okt()
        self.tokenizer = tokenizer

        print("[INIT] DeleteStopword...")

        # stopword 리스트 파일을 참조하여 불용어 리스트를 만듭니다.
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            temps = f.readlines()
            for word in temps:
                new_word = word.strip()
                self.stopwords.append(tokenizer.encode(new_word)[1]) # 불용어 토큰 리스트를 생성합니다.
        print("Done!")

    def remove_stopwords(self, passage:str) -> str:
        """
            입력으로 들어온 스트링에서 불용어 리스트(self.stopwords)에 해당하는 토큰을 제거한 후 반환합니다.
            input: str
            output: str - 불용어가 제거된 스트링
        """
        # passage에서 불필요한 심볼 '\n'을 제거합니다.
        passage = passage.replace('\\n', '')
        
        #word_tokens = self.okt.morphs(passage)
        word_tokens = self.tokenizer.encode(passage)
        print(f'[raw text]: {passage}')
        print(f'[tokens]: {word_tokens}')
        
        result = [word for word in word_tokens if not word in self.stopwords]

        print(f'[result]: {self.tokenizer.decode(result)}')
        return self.tokenizer.decode(result[1:-1])

    def konlpy_okt(self, passage:str) -> str:
        """
            Okt (형태소 분석기)를 이용해 passage에서 불필요한 부사, 형용사를 제거합니다.
        """
        # passage에서 불필요한 심볼 '\n'을 제거합니다.
        passage = passage.replace('\\n', '')
        okted_passage = self.okt.pos(passage)

        # 부사와 형용사 추출
        #ad_words = []
        for word, pos in okted_passage:
            if pos == 'Adverb' or pos == 'Adjective':
                #ad_words.append(word)
                self.ad_stopwords.append(tokenizer.encode(word)[1])
        #print(ad_words)

        print(self.ad_stopwords)

        # 부사 및 형용사 제거
        #word_tokens = self.okt.morphs(passage)
        word_tokens = self.tokenizer.encode(passage)
        print(f'[raw text]: {passage}')
        print(f'[tokens]: {word_tokens}')
        
        result = [word for word in word_tokens if not word in self.ad_stopwords]

        print(f'[result]: {self.tokenizer.decode(result)}')
        return self.tokenizer.decode(result[1:-1])

    def find_answer_index(self, passage:str, answer:str) -> int:
        """
            passage에서 answer word의 시작 인덱스를 찾아 반환합니다. 이를 통해 새로운 데이터셋을 구축할 수 있습니다.
            return
                'answer_start' : passage에서 answer word가 시작되는 index입니다.
        """
        answer_start = passage.find(answer)

        return answer_start
        

            


############
### DEMO ###
############

tokenizer = AutoTokenizer.from_pretrained(
    "klue/roberta-large",
    use_fast=True,
    )

ds = DeleteStopword(tokenizer=tokenizer)



from datasets import load_from_disk
dataset = load_from_disk("../data/train_dataset/")

context = dataset['train'][0]['context']
answer = dataset['train'][0]['answers']['text'][0] # answer가 2개 이상인 경우는 없는듯

#ds.remove_stopwords(context)

r = ds.konlpy_okt(context)
print(r)

ds.find_answer_index(context, answer)