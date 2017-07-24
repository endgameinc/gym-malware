import numpy as np
from gym_malware.envs.utils import interface, pefeatures
from gym_malware.envs.controls import manipulate2 as manipulate

from gym_malware import sha256_train, sha256_holdout, MAXTURNS
from collections import defaultdict

from keras.models import load_model

ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}

from train_agent_chainer import create_acer_agent
import gym

def evaluate( action_function ):
    success=[]
    misclassified = []
    for sha256 in sha256_holdout:
        success_dict = defaultdict(list)
        bytez = interface.fetch_file(sha256)
        label = interface.get_label_local(bytez)
        if label == 0.0:
            misclassified.append(sha256)
            continue # already misclassified, move along
        for _ in range(MAXTURNS):
            action = action_function( bytez )
            print(action)
            success_dict[sha256].append(action)
            bytez = manipulate.modify_without_breaking( bytez, [action] )
            new_label = interface.get_label_local( bytez )
            if new_label == 0.0:
                success.append(success_dict)
                break
    return success, misclassified # evasion accuracy is len(success) / len(sha256_holdout)

import os
def get_latest_model_from(basedir):
    dirs = os.listdir(basedir)
    lastmodel = -1
    for d in dirs:
        try:
            if int(d) > lastmodel:
                lastmodel = int(d)
        except ValueError:
            continue

    assert lastmodel >= 0, "No saved models!"
    return os.path.join(basedir, str(lastmodel))

if __name__ == '__main__':
    # baseline: choose actions at random
    random_action = lambda bytez: np.random.choice( list(manipulate.ACTION_TABLE.keys()) )
    random_success, misclassified = evaluate( random_action )
    total = len(sha256_holdout) - len(misclassified) # don't count misclassified towards success

    ENV_NAME = 'malware-test-v0' 
    env = gym.make(ENV_NAME)

    fe = pefeatures.PEFeatureExtractor()
    def agent_policy(agent):
        def f(bytez):
            # first, get features from bytez
            feats = fe.extract( bytez )
            action_index = agent.act( feats ) 
            return ACTION_LOOKUP[ action_index ]
        return f

    agent = create_acer_agent(env)
    # pull latest stored model
    last_model_dir = get_latest_model_from('models/acer_chainer')
    agent.load( last_model_dir )
    success, _ = evaluate( agent_policy(agent) )

    agent_score = create_acer_agent(env)
    # pull latest stored model
    last_model_dir = get_latest_model_from('models/acer_score_chainer')
    agent_score.load( last_model_dir )

    score_success, _ = evaluate( agent_policy(agent) )

    print("Success rate of random chance: {}\n".format( len(random_success) / total ))
    print("Success rate (black box): {}\n".format( len(success) / total ) )
    print("Success rate (score): {}\n".format( len(score_success) / total ) )          


