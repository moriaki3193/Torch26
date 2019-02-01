from logging import getLogger

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

logger = getLogger(__name__)


class Lda(nn.Module):
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 num_steps=100, learning_rate=0.001, jit=True, use_cuda=False):
        super(Lda, self).__init__()
        self.corpus = corpus
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

        logger.info('-' * 40)
        logger.info(f'Training on {self.num_docs} documents')

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
