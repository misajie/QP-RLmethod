from collections import deque
import random
import torch
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def sample_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, nstate, done):
        experience = [state, action, reward, nstate, done]
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

def collate_fn(data):
    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    state = [torch.FloatTensor(s[0]) for s in data]
    action = torch.LongTensor([s[1] for s in data])
    reward = torch.LongTensor([s[2] for s in data])
    nstate = [torch.FloatTensor(s[3]) for s in data]
    done = torch.LongTensor([s[4] for s in data])

    seq_len, nseq_len = torch.LongTensor([f.shape[0] for f in state]), torch.LongTensor([f.shape[0] for f in nstate])
    state, nstate = pad_sequence(state, batch_first=True).float(), \
                    pad_sequence(nstate, batch_first=True).float()
    # features,labels = features.unsqueeze(-1),labels.unsqueeze(-1)
    state, nstate = pack_padded_sequence(state, seq_len, batch_first=True), \
                    pack_padded_sequence(nstate, nseq_len, batch_first=True)
    return state, action, reward, nstate, done
    
if __name__ == '__main__':
    from qpenv.ASenv import ASenv
    init_episode = 20
    env = ASenv(mpc_path="../benchmarks/normal/",buffer_size=100)
    memory = ReplayBuffer(100)
    for _ in range(init_episode):
        state = env.reset()
        while True:
            action = env.random_action(state)
            nstate, reward, done, _ = env.step(action)
            transition = dict(
                state=state,
                action=action,
                nstate=nstate,
                reward=reward,
                done=done
            )
            # print(transition)
            memory.add(**transition)
            state = nstate
            if done:
                break

    sample = memory.sample_batch(16)
    state, action, reward, nstate, done = collate_fn(sample)
