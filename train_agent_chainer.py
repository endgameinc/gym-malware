import numpy as np

import gym
import gym_malware

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl 

from chainerrl.action_value import DiscreteActionValue
from chainerrl import links
from chainerrl.agents import acer
from chainerrl.distribution import SoftmaxDistribution
from chainerrl import misc
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import q_functions
from chainerrl.replay_buffer import EpisodicReplayBuffer
from chainerrl import v_functions
from chainerrl.initializers import LeCunNormal


class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=[1024,256]):
        super(QFunction,self).__init__()
        net = []
        inpdim = obs_size
        for i,n_hid in enumerate(n_hidden_channels):
            net += [ ('l{}'.format(i), L.Linear( inpdim, n_hid ) ) ]
            net += [ ('norm{}'.format(i), L.BatchNormalization( n_hid ) ) ]
            net += [ ('_act{}'.format(i), F.relu ) ]
            inpdim = n_hid

        net += [('output', L.Linear( inpdim, n_actions) )]

        with self.init_scope():
            for n in net:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])

        self.forward = net


    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        for n, f in self.forward:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        return chainerrl.action_value.DiscreteActionValue(x)

def create_ddqn_agent(env):

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_func = QFunction(obs_dim, n_actions)

    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    # Set the discount factor that discounts future rewards.
    gamma = 0.95

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.Boltzmann()

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1000)

    # Chainer only accepts numpy.float32 by default, make sure
    # a converter as a feature extractor function phi.
    phi = lambda x: x.astype(np.float32, copy=False)

    # Now create an agent that will interact with the environment.
    # DQN agent as described in Mnih (2013) and Mnih (2015).
    # http://arxiv.org/pdf/1312.5602.pdf
    # http://arxiv.org/abs/1509.06461        
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=32, update_interval=1,
        target_update_interval=100, phi=phi)

    return agent


def create_acer_agent(env):
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = acer.ACERSeparateModel(
        pi=links.Sequence(
            L.Linear( obs_dim, 1024, initialW=LeCunNormal(1e-3)),
            F.relu,
            L.Linear( 1024, 512, initialW=LeCunNormal(1e-3)),
            F.relu,
            L.Linear( 512, n_actions, initialW=LeCunNormal(1e-3)),
            SoftmaxDistribution),
        q=links.Sequence(
            L.Linear( obs_dim, 1024, initialW=LeCunNormal(1e-3)),
            F.relu,
            L.Linear( 1024, 512, initialW=LeCunNormal(1e-3)),
            F.relu,
            L.Linear( 512, n_actions, initialW=LeCunNormal(1e-3)),
            DiscreteActionValue),
        )

    opt = rmsprop_async.RMSpropAsync( lr=7e-4, eps=1e-2, alpha=0.99)
    opt.setup( model )
    opt.add_hook( chainer.optimizer.GradientClipping(40) )

    replay_buffer = EpisodicReplayBuffer( 128 )
    agent = acer.ACER( model, opt, 
        gamma=0.95, # reward discount factor
        t_max=32, # update the model after this many local steps
        replay_buffer=replay_buffer,
        n_times_replay=4, # number of times experience replay is repeated for each update
        replay_start_size=64, # don't start replay unless we have this many experiences in the buffer
        disable_online_update=True, # rely only on experience buffer
        use_trust_region=True,  # enable trust region policy optimiztion
        trust_region_delta=0.1,  # a parameter for TRPO
        truncation_threshold=5.0, # truncate large importance weights
        beta=1e-2, # entropy regularization parameter
        phi= lambda obs: obs.astype(np.float32, copy=False) )

    return agent


def train_agent(rounds=10000, use_score=False, name='result_dir', create_agent=create_ddqn_agent):
    ENV_NAME = 'malware-score-v0' if use_score else 'malware-v0'
    env = gym.make( ENV_NAME ) 
    np.random.seed(123)
    env.seed(123)

    agent = create_agent(env)

    chainerrl.experiments.train_agent_with_evaluation(
        agent, env,
        steps=rounds,                   # Train the agent for this many rounds steps
        max_episode_len=env.maxturns,   # Maximum length of each episodes        
        eval_interval=1000,             # Evaluate the agent after every 1000 steps
        eval_n_runs=100,                # 100 episodes are sampled for each evaluation        
        outdir=name)                    # Save everything to 'result' directory

    return agent

if __name__ == '__main__':
    agent_score = train_agent( rounds=50000, use_score=True, name='models/acer_score_chainer', create_agent=create_acer_agent) # allow agent to see scores
    # models are automatically saved

    agent_blackbox = train_agent( rounds=50000, use_score=False, name='models/acer_chainer', create_agent=create_acer_agent) # black blox
    # models are automatically saved





