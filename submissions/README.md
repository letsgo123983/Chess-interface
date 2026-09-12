# Submissions

Every zip unpacks to `agent.py` at its root with `weights/` beside it, which
is what the platform and `.github/workflows/match.yml` expect. **The network
is identical in all of them** (`weights/net.npz` from the bot received on
10 September); the differences are search, time management and start-up.

| zip            | what it is | vs `base`, 15s+0.15s |
|----------------|------------|----------------------|
| `original.zip` | the bot received on 10 September, untouched. The one to beat. | -- |
| `a.zip`        | **= v8.zip**, the pick. What a push to this file sends to CI. | -- |
| `b.zip`        | = `original.zip`, the other side of the CI run. | -- |
| `base.zip`     | the 10 Sept submission plus the mate-score fix (stop on a mate only once the iteration is deep enough to have walked the line). Everything below builds on it. | -- |
| `v6.zip`       | base + the node limit re-tied to the clock before every depth, + warm-up deadline 78s -> 72s. | **53.6%** over 84 games (LOS 82%) |
| `v8.zip`       | v6 + more clock early (increment share 0.7 -> 0.85, cap 35% -> 40%), a harder cut once the best move has held four depths, and an instant reply when only one move is legal. | **54.3%** over 35 games (LOS 82%) |
| `v7.zip`       | v6 + singular extensions from depth 6 with double extensions, and history pruning of bad quiet moves near the leaves. | 51.7% over 30 games (LOS 59%) |
| `v2.zip`       | base + v6's changes + quiescence results stored in the table. **Rejected.** | 46.8% over 31 games |
| `v3.zip`       | v2 + v7's search changes. Carries v2's quiescence store. | 48.3% over 30 games |
| `v4.zip`       | v2 + v8's clock changes. Carries v2's quiescence store. | 48.4% over 31 games |
| `v5.zip`       | v3's search with v4's clock. Carries v2's quiescence store. Untested. | -- |

## What the numbers say

Storing quiescence results in the table is a **regression**: the three builds
carrying it (v2, v3, v4) scored 44/92 = 47.8% between them, while the same
changes without it are all above 50%. It is out of v6, v7 and v8.

Nothing else is separated by these runs. v6 and v8 are both around 54% with a
likelihood of superiority near 82%, which is suggestive and not proof -- the
interval still includes equality, and 15s+0.15s is a sixth of the real clock.
Confirming a margin this size needs roughly 400 games, which is what the
pipeline is for. v8 is the pick because it contains v6's changes and scored no
worse; v7's sharper search is the one to retest first if you want more.

## Running a match

Actions -> "bot match" -> Run workflow, and set `a` and `b` to any two file
names from this directory (`v3.zip` against `original.zip`, say). Defaults are
`a.zip` against `b.zip`, 50 book positions = 100 games at 120s+0.5s. The
standings, with Elo and a confidence interval, appear in the run summary.

Locally, without waiting for CI:

    python tools/local_match.py engines/v3 engines/base --pairs 25 --jobs 2 \
        --ms 15000 --inc 150 --out runs/x/results_0.csv
    python bot_arena/pool.py runs/x          # pools several runs into one verdict

Set `CHESS_WARMUP_DEADLINE_S=1000` locally: several engines compiling at once
on one machine is not a condition the competition has, and without the
override the deadline fires and the first move pays for it.

## Start-up

Compilation runs in a background thread and the import returns at the
deadline whatever happens, so the init budget is never overrun; anything left
is finished on our own clock at move one. Measured on an idle pinned core:
62.7s import + 2.1s on move one for `base`, ~59s for v2-v5.
