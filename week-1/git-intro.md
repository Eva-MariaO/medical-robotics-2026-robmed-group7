# Introduction to Git

Git records the history of a project as a sequence of commits. In this course, it also lets group members
work on the same repository and packages the submitted history into a single Moodle upload.

## Key concepts

| Term | Meaning |
|---|---|
| **Repository** | A project folder whose history is maintained by Git. |
| **Commit** | A saved snapshot with a message describing the change. |
| **Staging area** | The files selected for inclusion in the next commit. |
| **Branch** | A separate line of development within a repository. |
| **Fork** | A GitHub repository created from another repository. Each group uses one shared fork. |
| **Remote** | A repository hosted elsewhere. The group repository is `origin`; the course repository is `upstream`. |
| **Clone** | A local copy of a repository, including its history. |
| **Push / Pull** | Sending commits to the shared repository / receiving commits from it. |
| **Bundle** | A single file containing Git history, used for Moodle submission. |

## Create the shared group repository

There must be exactly **one working repository per group**, not one per student.

1. One member opens the course repository on GitHub and creates a fork named `robmed-group-XX`, replacing
   `XX` with the group number.
2. That member adds the remaining group members under **Settings → Collaborators**.
3. Every member accepts the invitation and clones the same group repository:

   ```bash
   git clone <group repository URL>
   cd <repository folder>
   ```

4. Every member adds the course repository as `upstream`:

   ```bash
   git remote add upstream https://github.com/LxViRaL-teaching/medical-robotics-2026.git
   git remote -v
   ```

`git remote -v` should show the group repository as `origin` and the course repository as `upstream`.

## Everyday workflow

Before starting work, receive your groupmates' latest commits:

```bash
git pull
```

Inspect and save your work in small, meaningful commits:

```bash
git status
git add <files>
git commit -m "Describe the completed work"
git push
```

Pull before editing shared files, and commit and push regularly. A merge conflict occurs when Git cannot
automatically combine overlapping changes. If one occurs, do not force or discard anything; ask for help.

## Receive new course material

When the docente announces an update, one group member runs:

```bash
git switch main
git pull origin main
git fetch upstream
git merge upstream/main
git push origin main
```

The other members then receive the update with their normal `git pull`.

## Submitting your work on Moodle

Only one group member submits. First ensure that all work has been committed and pushed, then update the
local repository and check that it is clean:

```bash
git pull
git status
```

Uncommitted changes and untracked files are not included in a bundle. Create and verify the bundle,
replacing `XX` and `N` with the group and week numbers:

```bash
git bundle create robmed-group-XX-week-N.bundle --all
git bundle verify robmed-group-XX-week-N.bundle
```

Test it by cloning it outside the repository:

```bash
git clone robmed-group-XX-week-N.bundle ../bundle-check
```

Confirm that the expected files are present, then upload the `.bundle` file to the corresponding Moodle
activity. Do not submit a ZIP file or the `.git` directory. Moodle's upload time is the official submission
time; commit timestamps are not.
