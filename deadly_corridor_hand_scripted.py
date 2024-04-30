import time
from Environments import VizDoomGym
ge = VizDoomGym()

# idz do przodu
for i in range(3):
    state, reward, terminated, truncated, info = ge.step(3)
    print(reward)
    time.sleep(0.5)

# turn right
for i in range(2):
    state, reward, terminated, truncated, info = ge.step(6)
    print(reward)
    time.sleep(0.5)

# shoot
for k in range(2):
    state, reward, terminated, truncated, info = ge.step(2)
    print(reward)
    time.sleep(0.5)

# turn left
for k in range(4):
    state, reward, terminated, truncated, info = ge.step(5)
    print(reward)
    time.sleep(0.5)

# shoot
for i in range(2):
    state, reward, terminated, truncated, info = ge.step(2)
    print(reward)
    time.sleep(0.5)

# turn right
for i in range(3):
    state, reward, terminated, truncated, info = ge.step(6)
    print(reward)
    time.sleep(0.5)

# go forward
for i in range(12):
    state, reward, terminated, truncated, info = ge.step(3)
    print(reward)
    time.sleep(0.5)

# turn left
for i in range(2):
    state, reward, terminated, truncated, info = ge.step(5)
    print(reward)
    time.sleep(0.5)

# shoot
for i in range(2):
    state, reward, terminated, truncated, info = ge.step(2)
    print(reward)
    time.sleep(0.5)

# go forward
for i in range(3):
    state, reward, terminated, truncated, info = ge.step(3)
    print(reward)
    time.sleep(0.5)

# turn right
for i in range(3):
    state, reward, terminated, truncated, info = ge.step(6)
    print(reward)
    time.sleep(0.5)

# go forward
for i in range(4):
    state, reward, terminated, truncated, info = ge.step(2)
    print(reward)
    time.sleep(0.5)

# go forward
for i in range(7):
    state, reward, terminated, truncated, info = ge.step(5)
    print(reward)
    time.sleep(0.5)