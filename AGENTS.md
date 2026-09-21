# AI coding workflow

## Hard rule: GitHub first, Lovable never by default

- Treat this GitHub repository as the source of truth for code.
- Never use Lovable as a programming or code-generation agent unless the user explicitly asks for Lovable to write code.
- For implementations, bug fixes, refactors, settings changes, and small code edits, work directly in GitHub or through an approved coding environment such as Claude Code or Codex.
- Lovable may be used for preview, runtime checks, and publishing only. Do not send implementation prompts to Lovable merely because the project is connected to it.
- Before any code change, verify the repository, current branch, git status, and remotes.
- Respect repository-specific branch rules. Do not merge to or push directly to a protected or stable branch unless the user has explicitly authorized it.
- Never hardcode runtime secrets or credentials in GitHub. Keep secrets in runtime configuration.
- Avoid rewriting published Git history on Lovable-connected repositories.
- Minimize paid-agent usage. Do not spend Lovable tokens for work that can be done directly in GitHub.

Default assumption: GitHub first. Lovable only when explicitly requested for coding.
