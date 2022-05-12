from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large", # klue/bert-base, klue/roberta-large
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=5, # 10
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )

    train_retrieval: bool = field(
        default=True,
        metadata={
            "help": "Retreiver 훈련 & pkl"
        }

    )
    retrieval_type: str = field(
        default='dense',
        metadata={
            "help": "Define retrieval type (sparse, dense, bm25)"
        }

    )


@dataclass
class UserArguments:
    """
    WandB 관련 argument 입니다.
    """
    entity: str = field(
        default="boostcamp_nlp06",
        metadata={"help": "WandB의 entity name을 입력해주세요."
        }
    )

    name: str = field(
        default="klue/roberta-large_dense",
        metadata={"help": "WandB 상 표시될 실험 케이스의 이름입니다. 규칙에 따라 작성해 주세요."}
    )

@dataclass
class DenseTrainingArguments:
    """
    Arguments for training dense retrieval
    """
    data_path: str = field(
        default="./data/train_dataset",
        metadata={
            "help": "Train data path"
        },
    )
    dense_base_model: str = field(
        default="klue/roberta-small",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    dense_mode: str = field(
        default="single", #double
        metadata={
            "help": "single: share weight between p_encoder, q_encoder / double: not share"
        },
    )
    dense_passage_retrieval_name: str = field(
        default="./models/best/p_encoder",
        metadata={
            "help": "Path to pretrained model"
        },

    )
    dense_question_retrieval_name: str = field(
        default="./models/best/q_encoder",
        metadata={
            "help": "Path to pretrained model"
        },
    )
    dense_train_epoch: int = field(
        default=10,
        metadata={
            "help": "Epochs"
        },
    )
    dense_train_batch_size: int = field(
        default=8,
        metadata={
            "help": "batch size for train DataLoader"
        },
    )
    dense_train_learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": "learning_rate for training"
        },
    )
    dense_context_max_length: int = field(
        default=384,
        metadata={
            "help": "batch size for train DataLoader"
        },
    )
    dense_question_max_length: int = field(
        default=80,
        metadata={
            "help": "batch size for train DataLoader"
        },
    )
    dense_train_output_dir: str = field(
        default="./models/dense_train/",
        metadata={
            "help": "save directory"
        },
    )
    use_wiki_data: bool = field(
        default=True,
        metadata={
            "help": "Whether to use wiki data or not."
        },
    )
    wiki_data_path: str = field(
        default="./opt/ml/data/",
        metadata={
            "help": "Wiki data path"
        },
    )