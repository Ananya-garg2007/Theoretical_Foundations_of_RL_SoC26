from replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=5)

for i in range(5):

    state = [i, i + 1]

    action = i % 2

    reward = i

    next_state = [i + 1, i + 2]

    done = False

    buffer.push(
        state,
        action,
        reward,
        next_state,
        done
    )

print("Replay Buffer Size:")

print(len(buffer))

print()

print("Random Sample:")

sample = buffer.sample(3)

for transition in sample:

    print(transition)