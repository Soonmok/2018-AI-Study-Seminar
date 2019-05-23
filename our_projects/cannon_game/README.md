# KEEPING PABLO (GAME AI Agent)

# Overview
This is a project to implement Game AI agent playing simple a game.

This project use DQN algorithm and CNN network to get game environment information.

I took some code snippets of DQN algorithm from `yenchenlin` 's github project

# Play video

## 10 minutes later ..

## 1 hour later ..

# dependencies 
```
tensorflow
opencv
pygame
```

# Policy
crashed to triangle -> `-1 reward`

crashed to small dot (EndPoint) -> `1000 reward`

closer to small dot (EndPoint) -> `Phase 1 = 0.1 reward  /  Phase 2 = 0.5 reward`

# References
https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/README.md / yenchenlin . 
