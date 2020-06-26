__version__ = "1.2.0"
# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

# Tokenizer
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer

# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

# Modeling
from .modeling_utils import (PreTrainedModel, prune_layer, Conv1D)

from .modeling_bert import (BertPreTrainedModel, BertModel, BertForPreTraining,
                            BertForMaskedLM, BertForNextSentencePrediction,
                            BertForSequenceClassification, BertForMultipleChoice,
                            BertForTokenClassification, BertForQuestionAnswering,
                            load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                            BertForMedical, BertWithCAMLForMedical, 
                            BertTinyParallel3WithCAMLForMedical, BertTinyParallel4WithCAMLForMedical)
# Optimization
from .optimization import (AdamW, ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
                           WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule)

# Files and general utilities
from .file_utils import (PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME)

# Sparsemax
from .sparsemax import Sparsemax

from .replay_policy import ReplayPolicyNet
