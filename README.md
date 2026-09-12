# Chess bot match harness

Plays two Python chess agents against each other over a book of openings, each
position twice with colours reversed, and reports the score with an Elo
estimate and a confidence interval.

`submissions/a.zip` and `submissions/b.zip` are the two bots. Actions ->
"bot match" -> Run workflow runs them; a push to either zip runs them with the
defaults: 30 book positions = 60 games at 120s + 0.5s, over 20 parallel jobs
of three games. `a` and `b` inputs take any file name in `submissions/`.

Locally:

    pip install -r requirements.txt
    python tools/local_match.py engines/a engines/b --pairs 25 --jobs 2 \
        --ms 15000 --inc 150 --out runs/x/results_0.csv
    python bot_arena/pool.py runs/x
