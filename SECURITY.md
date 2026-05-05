# Security

Please do not report security issues through public GitHub issues.

For this research codebase, the main release risks are accidental publication of
credentials, private artifact locations, checkpoints, or dataset shards. Before
sharing changes, scan for secrets and machine-local paths.

Recommended local scan:

```bash
rg -n "AWS_SECRET|AWS_ACCESS_KEY|HF_TOKEN|HUGGINGFACE_HUB_TOKEN|WANDB_API_KEY|BEGIN .*PRIVATE KEY" .
rg -n "/Users/|/private/|/Volumes/" README.md docs configs scripts ttt tests pyproject.toml
```
