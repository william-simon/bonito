"""
Bonito CTC-CRF Model.
"""

import torch
import numpy as np

import koi
from koi.ctc import SequenceDist, Max, Log, semiring
from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse

from bonito.nn import Module, Convolution, LinearCRFEncoder, Serial, Permute, layers, from_dict


def get_stride(m):
    children = list(m.children())
    if len(children) == 0:
        if hasattr(m, "stride"):
            stride = m.stride
            if isinstance(stride, int):
                return stride
            return np.prod(stride)
        return 1
    return np.prod([get_stride(c) for c in children])


def twoD_softmax(mat):
    return torch.flatten(mat,start_dim=-2).softmax(-1).reshape(*mat.shape)

class CTC_CRF(SequenceDist):

    def __init__(self, state_len, alphabet, n_pre_context_bases=0, n_post_context_bases=0, probs_method="ont", traceback_method="ont"):
        super().__init__()
        print(f'{probs_method}')
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_pre_context_bases = n_pre_context_bases
        self.n_post_context_bases = n_post_context_bases
        self.n_base = len(alphabet[1:])
        self.num_rows = self.n_base**self.state_len
        num_trans = self.num_rows * (self.n_base+1) # +1 for the blank value
        self.idx = torch.cat([
            torch.arange(self.num_rows)[:, None],
            torch.arange(self.num_rows
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int64)
        next_trans_idx = torch.arange(0,num_trans).reshape(-1,self.n_base + 1)[:,1:]
        next_blank_idx = torch.arange(0,num_trans, self.n_base + 1).reshape(self.n_base, -1).T
        self.next_idx = torch.cat([next_trans_idx[i] if i%self.n_base != 0 else \
                                torch.cat([next_blank_idx[i//self.n_base],next_trans_idx[i]]) \
                                for i in range(next_trans_idx.shape[0])]).reshape(-1,self.n_base).to(torch.int64)
        t = torch.arange(self.num_rows)
        self.next_reorder = torch.cat([t[i::4] for i in range(4)])
        self.probs_method, self.probs_lookAhead = probs_method.split('_') if probs_method != "ont" else ("ont",-1)
        self.traceback_method, self.traceback_lookAhead = traceback_method.split('_') if traceback_method != "ont" else ("ont",-1)
    def n_score(self):
        return len(self.alphabet) * self.num_rows

    def logZ(self, scores, S:semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.num_rows), S.one)
        beta_T = Ms.new_full((N, self.num_rows), S.one)
        return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.num_rows), S.one)
        return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.num_rows), S.one)
        return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat([
            log_trans_probs[:, :, :, [0]],
            log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
        ], dim=-1)
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        #convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(scores[..., 0].permute(
            0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
        )
        emissions = torch.flip(scores[..., 1:].permute(
            0, 1, *range(self.state_len, 1, -1),
            self.state_len +2,
            self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def chain_scores(self, inputs):
        # with torch.no_grad():
            L, B, NR, NB = inputs.shape[0], inputs.shape[1], self.num_rows, self.n_base
            inputs = inputs.detach().reshape(L, B, NR, NB+1)
            probs = inputs.clone()
            cs = torch.zeros_like(inputs)
            lhSoft = int(self.probs_lookAhead)
            lhMax = int(self.traceback_lookAhead)
            idx = self.idx.to(inputs.device).detach()
            next_idx = self.next_idx.to(inputs.device).detach()
            num_trans = NR * (NB+1) # +1 for the blank value
            for i in range(1,B):
                idx = torch.concat((idx,idx[0:NR] + i*NR))
                next_idx = torch.concat((next_idx,next_idx[0:NB+1] + i*num_trans))
            for i in range(1,L):
                l = torch.take(probs[i-1].logsumexp(-1),idx).reshape(B,NR,NB+1)
                probs[i] += l - l.min(-1)[0].min(-1)[0].reshape(-1,1,1)
                if i >= lhSoft:
                    r = torch.take(probs[i],next_idx).reshape(-1,NB+1,NB).logsumexp(1).unsqueeze(1).permute(0,2,1)
                    for j in range(1,lhSoft):
                        r = torch.take(probs[i-j]+r-r.min(1)[0].unsqueeze(1),next_idx).reshape(-1,NB+1,NB).logsumexp(1).unsqueeze(1).permute(0,2,1)
                    probs[i-lhSoft] +=  r
                    cs[i-lhSoft] = (twoD_softmax(probs[i-lhSoft])+1e-7).log()
                    if i > lhSoft:
                        l = torch.take(cs[i-lhSoft-1].max(-1)[0],idx).reshape(B,NR,NB+1)
                        cs[i-lhSoft] += l - l.min(-1)[0].min(-1)[0].reshape(-1,1,1)
                        if i >= lhSoft + lhMax:
                            r = torch.take(cs[i-lhSoft],next_idx).reshape(-1,NB+1,NB).max(1)[0].unsqueeze(1).permute(0,2,1)
                            for j in range(1,lhMax):
                                r = torch.take(cs[i-lhSoft-j] + r - r.min(1)[0].unsqueeze(1),next_idx).reshape(-1,NB+1,NB).max(1)[0].unsqueeze(1).permute(0,2,1)
                            cs[i-lhSoft-lhMax] += r
            # for i in range(1,lhSoft): # There is uncompleted code here for finishing the chain score
            #     probs[L-i-1] = probs[L-i] + torch.take(probs[L-i],next_idx).reshape(-1,NB+1,NB).logsumexp(1).T.reshape(-1,1)
            return cs.reshape(L,B,-1)

    def viterbi(self, scores):
        if self.traceback_method == 'ont':
            traceback = self.posteriors(scores, Max)
        elif self.traceback_method == 'chain':
            traceback = self.chain_scores(scores)
        a_traceback = traceback.argmax(-1)
        moves = (a_traceback % len(self.alphabet)) != 0
        paths = 1 + (torch.div(a_traceback, len(self.alphabet), rounding_mode="floor") % self.n_base)
        return torch.where(moves, paths, 0)

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * (self.n_base ** (self.state_len - i - 1))
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = - (logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        else:
            raise ValueError('Unknown reduction type {}'.format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)


def conv(c_in, c_out, ks, stride=1, bias=False, activation=None, norm=None):
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation, norm=norm)


def rnn_encoder(n_base, state_len, insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True, num_layers=5, norm=None):
    rnn = layers[rnn_type]
    return Serial([
        conv(insize, 4, ks=5, bias=True, activation=activation, norm=norm),
        conv(4, 16, ks=5, bias=True, activation=activation, norm=norm),
        conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation, norm=norm),
        Permute([2, 0, 1]),
        *(rnn(features, features, reverse=(num_layers - i) % 2) for i in range(num_layers)),
        LinearCRFEncoder(
            features, n_base, state_len, activation='tanh', scale=scale,
            blank_score=blank_score, expand_blanks=expand_blanks
        )
    ])


class SeqdistModel(Module):
    def __init__(self, encoder, seqdist, n_pre_post_context_bases=None):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet
        if n_pre_post_context_bases is None:
            self.n_pre_context_bases = self.seqdist.state_len - 1
            self.n_post_context_bases = 1
        else:
            self.n_pre_context_bases, self.n_post_context_bases = n_pre_post_context_bases

    def forward(self, x, *args):
        return self.encoder(x)

    def decode_batch(self, x):
        if self.seqdist.probs_method == 'ont':
            scores = (self.seqdist.posteriors(x.to(torch.float32)) + 1e-8).log()
        elif self.seqdist.probs_method == "chain":
            scores = x
        else:
            raise ValueError(f'Unknown decode method {self.seqdist.probs_method}')
        tracebacks = self.seqdist.viterbi(scores).to(torch.int16).T
        bases = [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]
        return bases
    

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]

    def loss(self, scores, targets, target_lengths, **kwargs):
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs)

    def use_koi(self, **kwargs):
        self.encoder = koi.lstm.update_graph(
            self.encoder,
            batchsize=kwargs["batchsize"],
            chunksize=kwargs["chunksize"] // self.stride,
            quantize=kwargs["quantize"],
        )


class Model(SeqdistModel):

    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels'],
            probs_method='ont' if 'probs_method' not in config else config['probs_method'],
            traceback_method='ont' if 'traceback_method' not in config else config['traceback_method'],
        )
        if 'type' in config['encoder']: #new-style config
            encoder = from_dict(config['encoder'])
        else: #old-style
            encoder = rnn_encoder(seqdist.n_base, seqdist.state_len, insize=config['input']['features'], **config['encoder'])

        super().__init__(encoder, seqdist, n_pre_post_context_bases=config['input'].get('n_pre_post_context_bases'))
        self.config = config
