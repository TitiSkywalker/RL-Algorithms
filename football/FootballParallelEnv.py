"""
To boost performance, all environments are parallel. This is done by sending and receiving control signals from pipes.
"""

import torch
import torch.multiprocessing as mp

from FootballEnv import FootballSingleEnv


def worker(frames, skip_frames, pipe):
    """Worker process that manages an environment instance."""
    env = FootballSingleEnv(frames, skip_frames)
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "step":
                parameter, minimap, reward, done, info = env.step(data)
                pipe.send((parameter, minimap, reward, done, info))
            elif cmd == "reset":
                parameter, minimap = env.reset()
                pipe.send((parameter, minimap))
            elif cmd == "close":
                pipe.close()
                env.close()
                break
            elif cmd == "score_board":
                score_board = env.score_board
                pipe.send(score_board)
    except:
        pipe.close()
        env.close()


class FootballParallelEnv:
    def __init__(self, num_envs, frames=4, skip_frames=4):
        self.num_envs = num_envs

        self.workers = []
        self.pipes = []

        for index in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            process = mp.Process(target=worker, args=(frames, skip_frames, child_conn))
            process.start()
            self.workers.append(process)
            self.pipes.append(parent_conn)

    def step(
        self, actions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[bool], dict]:
        for pipe, action in zip(self.pipes, actions):
            pipe.send(("step", action))
        results = [pipe.recv() for pipe in self.pipes]
        parameters, minimaps, rewards, terminated, info = zip(*results)
        return (
            torch.stack(parameters),
            torch.stack(minimaps),
            torch.Tensor(rewards),
            terminated,
            info,
        )

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        for pipe in self.pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self.pipes]

        parameters = []
        minimaps = []
        for parameter, minimap in results:
            parameters.append(parameter)
            minimaps.append(minimap)
        return torch.stack(parameters), torch.stack(minimaps)

    def score_board(self):
        for pipe in self.pipes:
            pipe.send(("score_board", None))
        results = [pipe.recv() for pipe in self.pipes]

        return torch.Tensor(results).sum(dim=0).to(int).tolist()

    def close(self):
        for pipe in self.pipes:
            pipe.send(("close", None))
        for process in self.workers:
            process.join()
            process.close()
