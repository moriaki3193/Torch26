from contextlib import contextmanager
from logging import getLogger
from itertools import product
import re
import sys
import time
import unicodedata
import urllib3

from gensim import corpora
import MeCab
import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

logger = getLogger(__name__)


@contextmanager
def timer(msg):
    start_time = time.time()
    logger.info(msg)
    yield
    elapsed_time = time.time() - start_time
    if elapsed_time > 60:
        logger.info(f'done in {elapsed_time / 60:.2f} min.')
    else:
        logger.info(f'done in {elapsed_time:.2f} sec.')


class Tokenizer:
    def __init__(self):
        mecab = MeCab.Tagger('-d /usr/local/mecab/lib/mecab/dic/mecab-ipadic-neologd/')
        mecab.parse('')
        self.mecab = mecab
        self.parts = {'名詞', '動詞', '形容詞'}
        self.stop_words = self.make_stop_words()

    @staticmethod
    def make_stop_words():

        def urlopen(url):
            http = urllib3.PoolManager()
            res = http.request('GET', url)
            words = res.data.decode('utf-8').split()
            return words

        url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/'
        url_ja = url + 'Japanese.txt'
        url_en = url + 'English.txt'

        stop_words_ja = urlopen(url_ja)
        stop_words_en = urlopen(url_en)
        stop_words = stop_words_ja + stop_words_en
        stop_words += [chr(i) for i in range(12353, 12436)]  # ひらがな1文字
        stop_words += [chr(i) + chr(j) for i, j in product(range(12353, 12436), range(12353, 12436))]  # ひらがな2文字

        return set(stop_words)

    def extract_parts(self, text):
        mecab = self.mecab
        parts = self.parts
        stop_words = self.stop_words
        keywords = []
        node = mecab.parseToNode(text).next
        while node:
            if node.feature.split(',')[0] in parts and node.surface not in stop_words:
                keywords.append(node.surface)
            node = node.next
        return keywords

    def tokenize(self, text):
        text = unicodedata.normalize('NFKC', text)  # カナ全角、アルファベット半角にする
        text = re.sub(r'https?://[\w/:%#$&?()~.=+\-…]+', '', text)  # URLを削る
        text = re.sub(r'[0-9]+年|[0-9]+月|[0-9]+日|[0-9]+時|[0-9]+分', '', text)  # 日時を削る
        text = re.sub(r'[!-@[-`{-~]', '', text)  # 記号 + 半角数字 を削る
        text = text.lower()  # アルファベットを小文字にする

        return self.extract_parts(text)


class Lda(nn.Module):
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 num_steps=100, learning_rate=0.001, jit=True, use_cuda=False):
        super(Lda, self).__init__()
        self.id2word = id2word

        self.num_topics = num_topics
        self.num_words = len(id2word)
        self.num_docs = len(corpus)
        self.num_words_of_doc = [len(text) for text in corpus]

        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.jit = jit
        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()

        if corpus is not None:
            self.update(corpus)

    def update(self, corpus):
        logger.info('-' * 40)
        logger.info(f'Training on {self.num_docs} documents')

        self.corpus = corpus

        # setup the optimizer
        self.optim = Adam({'lr': self.learning_rate})

        # setup the inference algorithm
        Elbo = JitTraceEnum_ELBO if self.jit else TraceEnum_ELBO
        self.elbo = Elbo()
        self.svi = SVI(self.model, self.guide, self.optim, self.elbo)

        # train
        for step in range(self.num_steps):
            loss = self.svi.step()

            if step % 10 == 0:
                logger.info(f'Step {step: >5d}\tLoss: {loss}')

    def model(self):
        with pyro.plate('topics', self.num_topics):
            phi = pyro.sample('phi',
                              dist.Dirichlet(torch.ones(self.num_words) / self.num_words))

        for ind in range(self.num_docs):
            data = self.corpus[ind]
            num_words = self.num_words_of_doc[ind]
            theta = pyro.sample(f'theta_{ind}',
                                dist.Dirichlet(torch.ones(self.num_topics) / self.num_topics))

            with pyro.plate(f'words_{ind}', num_words):
                z = pyro.sample(f'z_{ind}', dist.Categorical(theta))
                w = pyro.sample(f'w_{ind}', dist.Categorical(phi[z]),
                                obs=data)

        return phi, theta, w

    def guide(self):
        phi_posterior = pyro.param(
            'phi_posterior',
            lambda: torch.ones(self.num_topics, self.num_words),
            constraint=constraints.greater_than(0.5))
        with pyro.plate('topics', self.num_topics):
            pyro.sample('phi', dist.Dirichlet(phi_posterior))

        theta_posterior = pyro.param(
            'theta_posterior',
            lambda: torch.ones(self.num_docs, self.num_topics),
            constraint=constraints.greater_than(0.5))
        with pyro.plate('documents', self.num_docs):
            pyro.sample('theta', dist.Dirichlet(theta_posterior))


def lda_train(texts):
    with timer('preprocessing...'):
        t = Tokenizer()
        word_list = [t.tokenize(text) for text in texts]
        dictionary = corpora.Dictionary(word_list)
        dictionary.filter_extremes()
        corpus = [dictionary.doc2idx(text) for text in word_list]

    with timer('LDA training...'):
        lda = Lda(corpus=corpus,
                  id2word=dictionary,
                  num_topics=10)

    return lda
