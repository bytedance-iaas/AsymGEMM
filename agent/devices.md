# Device mapping: gb200 tunnel names ↔ real machine codes

The `gb200-kevin-NN` names are **VS Code Remote Tunnel names** (started by `tunnel_start`
in `~/env/bashrc.sh`, kept alive by crontab), *not* SSH hostnames. The tunnel number `NN`
is **not** the node number `cZZ` — it is the **last octet of the node's 10.78.202.x IP**.

- Canonical name order is `gb200-kevin-NN` (not `gb200-NN-kevin`).
- Node-to-node SSH: `ssh <real hostname>` works passwordless as user `kevinni`.
  `ssh gb200-kevin-NN` does **not** resolve (tunnel names are not in DNS).
- Browser access to a tunnel: `https://vscode.dev/tunnel/<tunnel name>`

| Tunnel name      | Real hostname (device code) | IP            | Notes                     |
|------------------|-----------------------------|---------------|---------------------------|
| `gb200-kevin-40` | `s04-p1-dgx-02-c12`         | 10.78.202.40  |                           |
| `gb200-kevin-42` | `s04-p1-dgx-02-c14`         | 10.78.202.42  |                           |
| `gb200-kevin-45` | `s04-p1-dgx-02-c17`         | 10.78.202.45  | current primary work node |

All three rows verified live on 2026-08-14: on each node the tunnel process
(`code tunnel --name <name>`), the crontab `tunnel_start <name>` self-heal line, and
`$TUNNEL_DIR/<name>/` all agree (`TUNNEL_DIR=/scratch_local/user_data/kevinni/tunnels`
on the ByteDance/GB200 domain; UNITES/playpen machines use `/playpen/kevin/tunnels`
and are not part of this table).

To re-verify a row or add a new node:

```bash
ssh -o BatchMode=yes <real-hostname> \
  'hostname; ls /scratch_local/user_data/kevinni/tunnels; crontab -l | grep -o "tunnel_start [a-zA-Z0-9-]*"'
```

## Reaching the nodes from outside the cluster

All SSH logins arrive via the internal bastion/login host **10.78.200.8**; the
`s04-*` hostnames and `10.78.202.x` IPs resolve/route only inside the cluster network.
From a laptop you need whatever gets you onto that network first (VPN / corp SSH entry
to 10.78.200.8) — this doc cannot know your laptop-side credentials. Once the bastion
is reachable, either hop manually or use:

```bash
ssh -J <you>@10.78.200.8 kevinni@s04-p1-dgx-02-c17
```

Quick test that your route works: `ssh kevinni@s04-p1-dgx-02-c17 hostname` (however you
get there) must print `s04-p1-dgx-02-c17`. `/home` is one NFS export
(`10.78.200.27:/data/home`) shared by all nodes, so any one node sees all code+artifacts.

## Backing up / syncing to a local machine

The canonical, self-contained runbook is **`agent/impls/local_backup.md`** — hand that ONE file
to the agent on the destination machine. It contains the access details above plus the
wave structure (wave 1 source code = the 5 locally-modified repos WITH `.git`; wave 2
current artifacts = `env/`; optional waves 3–4 = LF datasets 84G / run history 455G),
the symlink layout rule (mirror both trees under one parent; never `rsync -L`), the
clean-repo re-clone manifest (`env/agent/third_party_patches/MANIFEST.tsv`), and the
verification steps. Sizes: ~20G core, ~560G everything.
