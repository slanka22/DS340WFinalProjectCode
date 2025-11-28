# Imports
import numpy as np
import pandas as pd
import torch
import subprocess

import os
import sys
from pathlib import Path
import yaml
import cv2

sys.path.append("C:\\Users\\Slank\\OneDrive\\Desktop\\TrafficDojo\\TrafficDojo\\dreamerV2-pytorch")
sys.path.append("C:\\Users\\Slank\\OneDrive\\Desktop\\TrafficDojo\\TrafficDojo\\dreamerV2-pytorch\\common")
sys.path.append("C:\\Users\\Slank\\OneDrive\\Desktop\\TrafficDojo\\TrafficDojo\\dreamerV2-pytorch\\dreamerv2")
from common import replay
from dreamerv2.agent import Agent
from meta_traffic.metatraffic_env.traffic_signal_env import MetaSUMOEnv
from meta_traffic.metatraffic_env.observation import TopDownObservation

#Run Data Generation
subprocess.run([
    "python",
    "world_models\\data\\generation_script.py",
    "--rollouts", "200",
    "--threads", "4",
    "--rootdir", "data\\generatedData",
    "--policy", "brown"
])

#Read TrafficDojo Data Generation
def readData(dreamerMemory):
    for i in range(4):
        threadFolder = "thread_" + str(i)
        currentFolder = os.path.join(".\\data\\generatedData\\", threadFolder)

        for dataFile in os.listdir(currentFolder):
            data = np.load(os.path.join(currentFolder, dataFile), mmap_mode="r")

            observations = data["observations"]
            observations = observations.reshape(-1, *observations.shape[-3:])
            images = observations.transpose(0, 3, 1, 2)

            actions = data["actions"].reshape(-1)
            correctedActions = np.eye(4, dtype=np.float32)[actions]

            dataInstance = {
                "image": images[:-1],
                "action": correctedActions[:-1],
                "reward": data["rewards"][:-1].astype(np.float32),
                "discount": (1.0 - data["terminals"][:-1]).astype(np.float32),
            }
            dreamerMemory.add(dataInstance)

trainDreamerReplay = replay.Replay(Path("replayBuffer"))
readData(trainDreamerReplay)

#Train Dreamer
class emptyEnv:
    def __init__(self, n): self.n = n

modelSpace = emptyEnv(4)

with open("C:\\Users\\Slank\\OneDrive\\Desktop\\TrafficDojo\\TrafficDojo\\dreamerV2-pytorch\\dreamerv2\\configs.yaml", "r") as f:
    dreamerConfig = yaml.load(f, Loader=yaml.FullLoader)["defaults"]

class configToDreamerExpectation(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

dreamerConfig["actor"]["dist"] = "onehot"
dreamerConfig["image_size"] = (64, 64)
dreamerConfig["grayscale"] = True
dreamerConfig = configToDreamerExpectation(dreamerConfig)

dreamerTrainer = Agent(dreamerConfig, logger=None, actspce=modelSpace, step=None).to(dreamerConfig["device"])

dataLoader = trainDreamerReplay.dataset(
    batch=16,
    length=30,
    oversample_ends=False,
)
dreanerDataset = iter(dataLoader)
for i in range(50000):
    currentBatch = next(dreanerDataset)
    for key in list(currentBatch.keys()):
        currentBatch[key] = torch.as_tensor(currentBatch[key], device=dreamerConfig.device)

    currentBatch["image"] = currentBatch["image"].permute(0, 1, 3, 2, 4).contiguous()
    torch.xpu.synchronize()

    trainOutput, metricsOutput = dreamerTrainer.train(currentBatch)

torch.save(dreamerTrainer.state_dict(), "FinalDreamerV2Model.pth")

#Evaluate Dreamer
device = dreamerConfig.device

dreamerModel = Agent(dreamerConfig, logger=None, actspce=modelSpace, step=None).to(device)
dreamerModel.load_state_dict(torch.load("FinalDreamerV2Model.pth", map_location=device))

dreamerModel._should_expl = lambda step: False
dreamerModel.expl_behavior_step = 0
dreamerModel.step = 0
dreamerModel._step = 0

env = MetaSUMOEnv(
    dict(
        image_on_cuda=False,
        begin_time=25200,
        delta_time=5,
        window_size=(256, 256),
        stack_size=1,
        min_green=10,
        sumo_cfg_file="C:\\Users\\Slank\\OneDrive\\Desktop\\TrafficDojo\\TrafficDojo\\nets\\RESCO\\cologne1\\cologne1.sumocfg",
        sumo_gui=False,
        capture_all_at_once=True,
        vision_feature=True,
        sumo_feature=True,
        agent_observation=TopDownObservation,
        top_down_camera_initial_z=100,
    ),
    num_seconds=6000,
    out_csv_name="dreamerEvalvuationOutput",
)

def preprocessImages(sumoImage):
    trafficImage = np.squeeze(sumoImage["image"])

    if trafficImage.ndim == 3:
        trafficImage = cv2.cvtColor(trafficImage, cv2.COLOR_BGR2GRAY)

    trafficImage = cv2.resize(trafficImage, (64, 64))
    return {"image": trafficImage[:, :, None]}

sumoImages, sumoOutputs = env.reset()
processedImage = preprocessImages(sumoImages)

currentDreamerStep = 0

def evalDreamer(processedObservations, agentState):
    global currentDreamerStep

    processedImage = processedObservations["image"].astype(np.float32) / 255.0
    processedImage = torch.from_numpy(processedImage.transpose(2, 0, 1)[None]).to(device)

    resetCondition = 1.0 if currentDreamerStep == 0 else 0.0
    dreamerInputs = {
        "image": processedImage,
        "reset": torch.tensor([[resetCondition]], device=device),
        "reward": torch.zeros((1, 1), device=device),
        "discount": torch.ones((1, 1), device=device),
        "action": torch.zeros((1, 4), device=device)
    }

    with torch.no_grad():
        dreamerOutputs, newState = dreamerModel.policy(dreamerInputs, agentState)

    currentDreamerStep += 1
    return int(dreamerOutputs["action"].argmax().cpu()), newState

metricList = []
agentState = None

while True:
    dreamerAction, agentState = evalDreamer(processedImage, agentState)

    rawObservation, reward, terminated, truncated, info = env.step(dreamerAction)
    processedImage = preprocessImages(rawObservation)

    metricList.append(info)

    if terminated or truncated:
        break

pd.DataFrame(metricList).to_csv("dreamerFinalMetrics.csv", index=False)
env.close()
