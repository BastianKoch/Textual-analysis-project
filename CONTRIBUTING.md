# Contributing to Textual Analysis Project

Thank you for contributing to this project! Please follow these guidelines to keep our codebase organized and maintainable.

## Getting Started

1. Make sure you've followed the setup instructions in [README.md](README.md)
2. Activate your virtual environment: `source venv/bin/activate`
3. Pull the latest changes: `git pull origin main`

## Branch Naming Convention

Create a new branch for each feature or fix:

```bash
git checkout -b feature/short-description
# or
git checkout -b bugfix/issue-description
# or
git checkout -b analysis/what-youre-analyzing
```

Examples:
- `feature/sentiment-analysis`
- `bugfix/transcript-parsing-error`
- `analysis/earnings-trends`

**Never commit directly to `main`** - always use a feature branch.

## Notebook Naming Convention

Follow this naming pattern for new notebooks:

```
{number}.{version}-{initials}-{short-description}.ipynb
```

Examples:
- `2.0-bk-sentiment-analysis.ipynb`
- `3.0-ad-feature-extraction.ipynb`
- `1.5-ar-data-exploration.ipynb`

**Guidelines:**
- Start with a number for ordering
- Use initials for tracking who worked on it
- Use lowercase and hyphens (not spaces)
- Make the description concise but clear

## Code Style

We use **ruff** for code formatting and linting. Always run these before committing:

```bash
# Check for issues (don't modify files)
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

Or use the shortcut:
```bash
make format    # Format all code
make lint      # Check without fixing
```

**Code style principles:**
- Write docstrings for all functions in `src/`
- Keep functions focused and single-purpose
- Use type hints where possible: `def load_transcript(filepath: str) -> str:`
- Add comments for complex logic

## Commit Messages

Write clear, descriptive commit messages:

```bash
# Good
git commit -m "Add sentiment analysis function to features module"
git commit -m "Fix transcript encoding issue in text_processing"
git commit -m "Update notebook with new cleaning parameters"

# Avoid
git commit -m "fix"
git commit -m "update"
git commit -m "WIP"
```

**Format:** Start with a verb (Add, Fix, Update, Remove), keep under 50 characters.

## Pull Request Process

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request (PR) on GitHub**
   - Give it a clear title
   - Describe what you changed and why
   - Reference any related issues

3. **Wait for review:**
   - At least one team member should review your code
   - Be open to feedback
   - Make requested changes

4. **Merge and delete branch:**
   ```bash
   # After approval, merge on GitHub
   # Then locally:
   git checkout main
   git pull origin main
   git branch -d feature/your-feature-name
   ```

## Working with Data

- ✅ **DO:** Add new data to `data/external/` (with docstrings explaining source)
- ❌ **DON'T:** Commit raw data files to Git (they're in `.gitignore`)
- ✅ **DO:** Document where external data comes from
- ✅ **DO:** Save processed outputs to `data/interim/` or `data/processed/`

## Working with Notebooks

- Use notebooks for **exploration and analysis**
- Extract reusable functions to `src/` modules
- Add markdown cells explaining what you're doing
- Clear outputs before committing: `Cell > All Output > Clear`
- Keep notebooks reasonably sized (easier to review)

**Example workflow:**
```python
# In your notebook
from src.text_processing import load_transcript, clean_text

# Explore and experiment
raw = load_transcript("data/raw/Transcripts/2003/sample.txt")
cleaned = clean_text(raw)
```

Then move reusable functions to `src/text_processing.py`.

## Adding Dependencies

If you need a new package:

1. Install it in your environment:
   ```bash
   pip install package-name
   ```

2. Add it to `requirements.txt`:
   ```bash
   # Edit the file and add your package
   ```

3. Commit the updated `requirements.txt`:
   ```bash
   git add requirements.txt
   git commit -m "Add new-package for feature-description"
   ```

Your teammates can then install it:
```bash
pip install -r requirements.txt
```

## Running Tests/Checks

Before pushing:

```bash
# Format and lint your code
make format
make lint

# Review your notebook outputs are cleared
# Check that your notebook runs without errors
```

## Asking Questions

- **Unclear requirements?** Ask in the group chat or comment on the issue
- **Need help?** @ mention teammates on GitHub or Slack
- **Found a bug?** Create an issue on GitHub with reproduction steps

## Common Issues

**"Permission denied" when pushing?**
```bash
git config credential.helper store
git push  # Enter credentials once, they'll be saved
```

**Want to undo a commit?**
```bash
git reset HEAD~1  # Undo last commit, keep changes
git revert HEAD   # Undo by creating new commit
```

**Accidentally committed to main?**
```bash
git reset --soft HEAD~1  # Undo, keep changes
git checkout -b feature/branch-name
git commit -m "your message"
git push origin feature/branch-name
```

---

**Questions?** Ask the team or check the [README.md](README.md)!
