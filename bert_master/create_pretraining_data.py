# coding=utf-8

"""Create masked LM/next sentence masked_lm TF examples for BERT.
  将原始输入语料转换成模型预训练所需要的数据格式TFRecoed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf

# 用于支持接受命令行传递参数，相当于接受argv
flags = tf.flags
FLAGS = flags.FLAGS

# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

#  一个句子里最多有多少个[MASK]标记
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

# 参数说明：对于同一个句子，我们可以设置不同位置的【MASK】次数。
# 比如对于句子Hello world, this is bert.，为了充分利用数据，第一次可以mask成Hello [MASK], this is bert.，
# 第二次可以变成Hello world, this is [MASK]
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

# 多少比例的Token被MASK掉
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

# 长度小于“max_seq_length”的样本比例。
# 因为在fine-tune过程里面输入的target_seq_length是可变的（小于等于max_seq_length），
# 那么为了防止过拟合也需要在pre-train的过程当中构造一些短的样本。
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
  """
  A single training instance (sentence pair).
  """
  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
    """
    args: tokens: 列表  masked tokens /是一句话
          segment_ids: 列表，元素是0或1，分别代表两句话
          masked_lm_positions: 被mask的词的索引
          masked_lm_labels: 被mask的词的原来的词
          is_random_next: 下一句是否随机选择
    """
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files):
  """
  function：Create TF example files from `TrainingInstance`s
  args:  instance:列表，元素是各个句子构成的instance(实例)
         tokenizer: 类实例，分词类
         output_files: 文件名列表
  """
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))    # 在writers中加入创建的.tf 文件

  writer_index = 0
  total_written = 0

  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)    # instance.tokens：list，元素是句子中的各个词，其中部分被mask;改代码
                                                                     # 表示将找出这句话中各个词在整个词表中的索引
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)     # 指的是在这句话中的索引位置
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)   # 此处指的是在所有词表中的索引
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    # 生成训练样本
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    # 输出到文件
    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)
    total_written += 1

    # 打印前20个样本
    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  # all_documents是list的list，第一层list表示document，
  # 第二层list表示document里的多个句子。
  all_documents = [[]]   # all_documents 最后形状[[[senten1],[sentence2]...(doucument1)],[[sentence1],[sentence2],...(document2)]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)   # 将all_documents 打乱
  vocab_words = list(tokenizer.vocab.keys())  # 词表

  instances = []
  for _ in range(dupe_factor):    # dupe_factor:同一个句子，可以设置不同位置mask的次数
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)   # instances存放instance实例，每个实例都代表一个被mask处理好的句子
  return instances

# 从一个文档中抽取多个训练样本
def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):

  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # 为[CLS], [SEP], [SEP]预留三个空位
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  # 以short_seq_prob的概率随机生成（2~max_num_tokens）的长度
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  # 注意document元素是列表，一个元素是一个句子
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          # 随机选取切分边界
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # 是否随机选择next
        is_random_next = False

        # 构建随机的下一句
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # 随机的挑选另外一篇文档的随机开始的句子
          # 但是理论上有可能随机到的文档就是当前文档，因此需要一个while循环
          # 这里只while循环10次，理论上还是有重复的可能性，但是我们忽略
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          # 上述构建句子时，current_chunk中句子并没有全用完，为避免浪费数据，将i跳回使用的句子后面
          i -= num_unused_segments

        # 构建真实的下一句
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        # 如果句子太长，将其截断
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        # 将两个句子合在一起，并加上特殊符号[cls,sep,sep]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        # 句子B结束加上[SEP]
        tokens.append("[SEP]")
        segment_ids.append(1)

        # 调用create_masked_lm_predictions来随机对某些Token进行mask
        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens, masked_lm_prob,
                                                                            max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
                                    tokens=tokens,
                                    segment_ids=segment_ids,
                                    is_random_next=is_random_next,
                                    masked_lm_positions=masked_lm_positions,
                                    masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):

  """
  functions: 创建 mask LM 数据
  args:   tokens: 一句话中词，标点等组成的列表
  return：
       output_tokens：一个列表，其中有些词（字）被mask替换
       masked_lm_positions： 列表，元素是output_tokens中被替换掉位置的索引（在当前句子中的索引）
       masked_lm_labels： 列表，元素是output_tokens中被替换成mask地方的原来的词
  """

  cand_indexes = []   # 存放一个句子中个个词的在当前句子中的索引，格式[[词1],[词2]，[##ci,##ci2]]
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)
  # round 四舍五入，默认四舍五入到整数
  # 此处计算一个句子中有多少个mask
  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []  # 里面存放namedtuple实例，实例的内容为（index:被mask的词在当前句子中索引；label:被mask的词（不是索引是实际的词））
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue

    # 将已经cover的词放入到列表中，便于下次检查
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  # 按索引大小排序
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)
  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # 句子太长，以0.5概率从头部去掉一些，以0.5的概率从尾部去掉些
    # 注意此处对trunc_tokes 操作就是对toke_a/b操作，注意列表的赋值
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    # tf.gfile.Glob(filename)查找匹配pattern的文件并以列表的形式返回，
    # filename可以是一个具体的文件名，也可以是包含通配符的正则表达式
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
                              input_files,
                              tokenizer,                          # 分词类的实例
                              FLAGS.max_seq_length,
                              FLAGS.dupe_factor,                  # 对于同一个句子，我们可以设置不同位置的【MASK】次数
                              FLAGS.short_seq_prob,               # 长度小于max_seq_length的样本比例
                              FLAGS.masked_lm_prob,               # 多少比例的Token被MASK掉  --15%
                              FLAGS.max_predictions_per_seq,      # 一个句子里最多有多少个[MASK]标记
                              rng)                                # 一个随机数

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  # 使用tf.flags.mark_flag_as_required(参数名)将对应的命令行参数标记为必需的,此时当没有设置对应的命令行参数时，会抛出异常信息
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")

  # 如果你的代码中的入口函数不叫main()，而是一个其他名字的函数，如test()，则你应该这样写入口tf.app.run(test)
  # 如果你的代码中的入口函数叫main()，则你就可以把入口写成tf.app.run()
  # 是主函数入口
  tf.app.run()
