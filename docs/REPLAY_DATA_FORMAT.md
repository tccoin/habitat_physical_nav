# `reference_replay` data format (Habitat-web ObjectNav human demos, MP3D)

Verified empirically over all 1141 episodes of `17DRP5sb8fy.json.gz`
(`python verify_replay_format.py <dataset.json.gz>`, 0 ordering violations).
Read this before indexing into `reference_replay` — the pairing is the opposite
of the intuitive one and is a natural source of off-by-one bugs.

## Structure

`data['episodes'][k]['reference_replay']` is a list of entries:

```python
{
  'action': 'TURN_RIGHT',              # see pairing rule below
  'agent_state': {
     'position': [x, y, z],            # world frame, metres; y is UP
     'rotation': [x, y, z, w],         # xyzw quaternion, world frame
     'sensor_data': {'rgb': {...}, 'semantic': {...}},  # sensor offsets, agent-local
  }
}
```

## Action / state pairing  ← the important part

**The `action` in entry `i` is the action that PRODUCED that entry's state** —
`replay[i]['agent_state']` is the state *after* `replay[i]['action']` executes.
The action is the *incoming* action, not the outgoing one; it owns the state
transition that ENDS at its entry, not the one that starts from it.

There is **one startup seam**, and it is universal (all 1141 episodes here):

- **Entry 0 is a seed:** `action == "STOP"` with the true initial state `s_0`.
  No action precedes it.
- **Entry 1 always duplicates entry 0's state** (`s_1 == s_0` exactly, in
  1141/1141 episodes). Entry 1 carries the human's *first real action*, but its
  state has not advanced yet — that action's effect first shows up in entry 2.
  So entry 1's action label is one frame ahead of its state. This is the only
  place the "action → resulting state" rule slips; from entry 2 on it is exact.

```
raw entries:  (STOP, s_0)  (a_1, s_0)  (a_2, s_2)  (a_3, s_3) ...  (STOP, s_N)
index:            0            1            2            3
                  ^seed        ^stale duplicate: state == s_0, action a_1 lags
```

Because of that seam, do NOT pair `(replay[i]['agent_state'], replay[i]['action'])`
as "state and the action taken from it" (off by one), and do NOT naively pair
state `i` with action `i+1` either (fails at the seam and at every action-type
boundary). Collapse the duplicate first, then the alignment is exact
(verified: 0 violations over 147,898 transitions):

```python
states  = [e["agent_state"] for e in replay]
actions = [e["action"]      for e in replay]
# Drop the stale duplicate startup frame at index 1.
states  = [states[0]]  + states[2:]
actions = [actions[0]] + actions[2:]
# Now for every i >= 1:  applying actions[i] to states[i-1] yields states[i].
#   (s, a, s') transitions:  (states[i-1], actions[i], states[i])
# actions[0] is the STOP seed; the final action is the terminal STOP.
```

Note: the raw action list overcounts real state changes by exactly one (the
phantom first action absorbed by the entry-1 duplicate); collapsing index 1
makes action and transition counts consistent.

`extract_path_3d` does `replay = replay[1:]`, which is sufficient for a
**position path** (entry 0 and entry 1 share the same position) but does NOT by
itself fix action↔state alignment — use the recipe above for `(s, a, s')` work.

## Action primitives (measured)

The action vocabulary is **six** actions (not four) across this dataset:

| Action         | Effect on state                          | Magnitude        | Stored in |
|----------------|------------------------------------------|------------------|-----------|
| `MOVE_FORWARD` | translate along local `-z`               | 0.25 m (less/0 on collision) | `position` |
| `TURN_LEFT`    | +30° yaw about world `+y`                | +30.0°           | `rotation` (base) |
| `TURN_RIGHT`   | -30° yaw about world `+y`                | -30.0°           | `rotation` (base) |
| `LOOK_UP`      | +30° camera pitch about local `x`        | +30.0°           | `sensor_data[*].rotation` |
| `LOOK_DOWN`    | -30° camera pitch about local `x`        | -30.0°           | `sensor_data[*].rotation` |
| `STOP`         | none (initial placeholder / episode end) | 0                | — |

Counts here: MOVE_FORWARD 89688, TURN_RIGHT 29650, TURN_LEFT 26995, LOOK_DOWN
889, LOOK_UP 676, STOP 2282. LOOK actions appear in 328/1141 episodes.

- Blocked forward moves (agent hits navmesh/wall) appear as `MOVE_FORWARD`
  entries with zero or partial displacement — expected, not an ordering error.
- `LOOK_UP`/`LOOK_DOWN` change **only** the camera pitch in
  `sensor_data['rgb'|'semantic']['rotation']` (base `position` and `rotation`
  are unchanged); the accumulated pitch **persists** across subsequent frames.
  So `sensor_data[*]['rotation']` is not always the identity `[0,0,0,1]` — it
  tracks the running look-tilt. A pure yaw/position analysis will (correctly)
  see a LOOK step as no change to the base pose.

## Coordinate convention

Standard Habitat: **right-handed, y-up, forward = local −z.**

- `position = [x, y, z]`, metres. **`y` is the vertical/up axis** (constant along
  a trajectory; `|Δy| ≈ 3e-6 m` during forward motion). The ground plane is `x–z`.
- `rotation` is an **xyzw** quaternion; heading (yaw) is rotation about `+y`:
  `yaw = atan2(2*(w*y + x*z), 1 - 2*(y*y + z*z))`.
- Forward (heading) direction in world frame = quaternion applied to `(0,0,-1)`.
- `sensor_data[*].position = [0, 0.88, 0]` is the camera height (0.88 m up),
  expressed in the agent-local frame — further confirming y-up.

> Note: the earlier informal description "x forward, y left, z up" is incorrect;
> the data is y-up with −z forward as above.
