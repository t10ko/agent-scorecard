# agent-scorecard

Measure what each Claude Code agent costs and delivers, then keep, fix, or remove it.

```console
$ agent-scorecard report --min-runs 5
Agent scorecard · ... → ... · 40 agent runs · agents $19.43 · main sessions $2.12

Group           Runs     Success    Cost  $/success  Tool errors          Verdict
web-researcher     6     2/6 33%   $2.06      $1.03           0%           remove
test-fixer         8     4/8 50%   $4.01      $1.00          10%              fix
code-writer       13   10/12 83%  $11.34      $1.13           2%             keep
Explore           10  10/10 100%   $1.23      $0.12           0%             keep
reviewer           3    3/3 100%   $0.79      $0.26           0%  not enough data
  └ remove: 2 of 6 decided runs succeeded (33%), below 50%
  └ fix: 4 of 8 decided runs succeeded (50%), below 80%
  └ keep: 10 of 12 decided runs succeeded (83%)
  └ 1 runs with unknown outcome
  └ keep: 10 of 10 decided runs succeeded (100%)
  └ not enough data: 3 decided runs, need 5

Prices checked 2026-09-23 against https://www.anthropic.com/pricing. Costs are API list prices.
```

(That is the real tool running on the synthetic demo data committed under
`examples/demo/`, so you can reproduce it byte for byte.)

## Why

Agent teams get expensive quietly: a fan-out that seemed cheap per turn bills every day,
across every session, and nothing in the normal workflow shows the total. Cost alone does not
say whether an agent is worth it, either — an expensive agent that delivers is a bargain, and a
cheap one that stalls is not. The session logs Claude Code already writes on your machine
contain enough to measure both, so this tool reads them where they are: nothing to install
inside your projects, nothing to wire up, nothing sent anywhere.

## Install

With [uv](https://docs.astral.sh/uv/):

```console
uvx --from git+https://github.com/t10ko/agent-scorecard agent-scorecard
```

or with pipx:

```console
pipx install git+https://github.com/t10ko/agent-scorecard
```

Python 3.12 or newer.

## Use

Run inside a project to score its agents (the logs are found under `~/.claude/projects/`):

```console
agent-scorecard report
```

Did an agent get better or worse on the new model? Group by agent type *and* model:

```console
agent-scorecard report --group-by agent-type+model
```

What actually failed? Drill into the runs:

```console
agent-scorecard runs --outcome failed
```

Where did the money go, per day, split into main sessions and agents?

```console
agent-scorecard cost --by-day
```

Feed another tool — the JSON schema is stable and sorted:

```console
agent-scorecard report --format json | jq '.groups[].verdict'
```

Other flags: `--transcripts DIR` to read a log folder directly (repeatable), `--since`/`--until`
for an inclusive UTC date window, `--prices FILE` to add or override rates, `-v` for the file
and line accounting, and `--success-pattern`/`--failure-pattern` to score runs by what the
agent's final report says.

## How it measures

**The logs it reads.** Claude Code writes one JSONL log per session under
`~/.claude/projects/<project>/`, plus one log and a `agent-<id>.meta.json` sidecar per subagent
under `<session>/subagents/`. This tool reads those files read-only — main-session logs from
the top level of the project folder, every subagent log under `subagents/` (including runs
nested under `workflows/`), and workflow `journal.jsonl` files for outcomes.

**De-duplication.** Claude Code writes one log line per content block of a single API turn, and
every one of those lines repeats the same `requestId`; subagent turns are often written twice
(a streamed partial, then the final count). Naively summing tokens overcounts, so the tool
keeps one record per `requestId` — the copy reporting the most billed tokens, which is the
final copy — and counts every discarded line by reason. If the counters cannot account for
every line read, or more than half the lines that should carry usage were lost, the tool
refuses to report rather than print a confident wrong number.

**Prices.** A bundled TOML table holds standard API list prices (checked against
[Anthropic's pricing page](https://www.anthropic.com/pricing)), including prompt caching:
cache reads at 0.1× the input rate, cache writes at 1.25× (5-minute TTL) and 2× (1-hour TTL).
Money is integer micro-dollars internally. `--prices FILE` adds models or replaces rates;
models with no row are reported as unpriced — never guessed, never shown as `$0`.

**How an outcome is decided.** A *run* is one agent started once. Its ending comes from the
first source that has one: the parent's foreground result, the parent's background task
notification, or the workflow journal; if none recorded it, the run's own file is consulted
(marked `inferred`), and runs with no signal at all are `unknown`. Then a run is:

1. `failed` if it was killed, stopped, or failed;
2. `failed` if `--failure-pattern` matches its final report;
3. `failed_tests` if it edited files, ran tests, and its last test run was red — a read-only
   agent investigating a failure is expected to see failing tests;
4. `succeeded` if it completed;
5. `succeeded` if it was `unknown` but `--success-pattern` matched its final report (marked
   inferred);
6. `unknown` otherwise — left out of every rate, always shown as a count.

**Verdicts.** Groups are sorted worst-first and given a verdict: `remove` when the success rate
falls below `--remove-below` (0.5), `fix` when it falls below `--fix-below` (0.8) or the tool
error rate exceeds `--max-tool-error-rate` (0.2), `keep` otherwise, and `not enough data` until
a group has `--min-runs` decided runs (5). **The verdict is a starting point for a human
decision, not an automatic kill switch.** The thresholds are judgment calls; the reasons and
notes printed under each row exist so you can overrule them.

## Limits

- **Test detection is a heuristic.** A run "ran tests" when a `Bash` command matched one of the
  built-in patterns (`pytest`, `go test`, `npm test`, `vitest`, `jest`, `cargo test`,
  `make test`, `mvn test`, `gradle test`, `dotnet test`, `rspec`, `phpunit`, `mix test`).
  Replace the whole list with `--test-command REGEX` (repeatable).
- **Some outcomes are inferred or unknown.** When the parent's record is gone, the run's own
  last turn decides and the result is marked `inferred`; runs with no signal count as
  `unknown` and are excluded from rates, not silently counted as successes or failures.
- **Prices need checking.** The bundled table ships with a `checked_on` date. Standard API
  rates only: no fast mode, batch, regional, or long-context pricing is modeled, and the
  official page is the source of truth.
- **Subscription users see API-equivalent dollars.** If you pay per subscription rather than
  per token, the figures are what the same usage would have cost on the API — useful for
  comparing agents, but not your bill.
- **The log format is undocumented and can change.** When it does, counters stop adding up and
  the tool refuses to report (exit code 1, with the accounting on stderr) rather than print
  wrong numbers.
- **Claude Code deletes old logs.** Local transcripts are kept for 30 days by default
  (`cleanupPeriodDays`); raise it if you want history for before-and-after comparisons.

## Privacy

- The tool only reads. It writes nothing unless you redirect its output, and it makes no
  network calls.
- Default output contains only agent types, model names, counts, costs, and reasons. Task
  descriptions appear only with `runs --show-descriptions`. Prompts and final reports never
  appear.
- Warnings never echo log-line content; a decode failure names the field that broke, not the
  line that broke it.

## Development

```console
uv sync                 # install with the dev tools
uv run pytest           # tests with coverage (>= 90% enforced)
uv run ruff check .     # lint
uv run ruff format --check .
uv run pyright          # strict mode for src/
uv run python scripts/make_demo.py   # regenerate examples/demo/
```

The demo output doubles as golden files: `tests/test_golden.py` fails if the CLI's output on
`examples/demo/` drifts, or if `make_demo.py` no longer reproduces the committed folder. After
an intentional output change, regenerate the demo and set
`AGENT_SCORECARD_UPDATE_GOLDEN=1` once to refresh the goldens.

## License

MIT. See [LICENSE](LICENSE).
