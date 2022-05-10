"""
    passage의 불용어를 제거하는 기능을 수행합니다. (query에는 적용하지 않음)

        input: passage
        output: stopword가 제거된 passage 반환

"""
from konlpy.tag import Okt # pip install tweepy==3.10.0 호환성 문제 있음
# apt install default-jdk
from transformers import AutoTokenizer

class DeleteStopword:
    def __init__(self, tokenizer, stopwords_path:str = './stopwords.txt') -> None:
        self.stopwords = []
        self.okt = Okt()
        self.tokenizer = tokenizer

        # stopword 리스트 파일을 참조하여 불용어 리스트를 만듭니다.
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            temps = f.readlines()
            for word in temps:
                new_word = word.strip()
                self.stopwords.append(tokenizer.encode(new_word)[1]) # 불용어 토큰 리스트를 생성합니다.
        #print(self.stopwords)

    def remove_stopwords(self, passage:str) -> str:
        """
            입력으로 들어온 스트링에서 불용어 리스트(self.stopwords)에 해당하는 토큰을 제거한 후 반환합니다.
            input: str
            output: str - 불용어가 제거된 스트링
        """
        
        #word_tokens = self.okt.morphs(passage)
        word_tokens = self.tokenizer.encode(passage)
        print(f'[raw text]: {passage}')
        print(f'[tokens]: {word_tokens}')
        
        result = [word for word in word_tokens if not word in self.stopwords]

        print(f'[result]: {self.tokenizer.decode(result)}')
        return result


############
### DEMO ###
############

tokenizer = AutoTokenizer.from_pretrained(
    "klue/roberta-large",
    use_fast=True,
    )

'''
print(tokenizer)
ts = tokenizer.encode("무궁화 꽃이 피었습니다. 대한민국 만세!")
print(ts)
for t in ts:
    print(tokenizer.decode(t))
'''

ds = DeleteStopword(tokenizer=tokenizer)

ds.remove_stopwords("우리집 강아지는 몹쓸 강아지네요. 바꾸어말하면 잘 생겼다. 네네치킨")
ds.remove_stopwords("고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지.")
#print(result)
