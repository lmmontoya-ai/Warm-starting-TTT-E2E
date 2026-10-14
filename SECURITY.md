# Security

Report security problems privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**. Please do not file a
public issue for them.

In this repository the realistic risk is a leaked credential, a private
storage location, or a dataset or checkpoint committed by mistake. If you spot
one, report it the same way.

Before you push, scan your changes:

```bash
git diff --cached | grep -nE "AKIA|K00[0-9][A-Za-z0-9+/]{27}|hf_[A-Za-z0-9]{30,}|WANDB_API_KEY=.+|BEGIN .*PRIVATE KEY"
```

Keep credentials in `.env`, which git ignores. `.env.example` lists the
variables the code reads.
