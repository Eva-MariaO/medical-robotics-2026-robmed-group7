# Week 1 — Environment Setup

The aim of this session is to prepare your computer for the practical work, establish your group's Git
workflow, and confirm that PyBullet runs correctly.

By the end of the session, you should have:

- Git and Miniconda installed;
- access to your group's shared repository;
- the `robmed` environment running locally;
- completed the introductory PyBullet notebook; and
- submitted a valid Git bundle through Moodle.

## 1. Install Miniconda

We use Miniconda to provide a consistent Python environment and a reliable cross-platform installation of
PyBullet.

**Windows** (PowerShell):

```powershell
winget install -e --id Anaconda.Miniconda3
```

**macOS** (with [Homebrew](https://brew.sh/)):

```bash
brew install --cask miniconda
```

Without Homebrew, download the Miniconda `.pkg` installer from
[anaconda.com/download](https://www.anaconda.com/download).

**Linux**:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Restart the terminal and check the installation:

```bash
conda --version
```

## 2. Install Git

**Windows** (PowerShell):

```powershell
winget install --id Git.Git -e --source winget
```

**macOS:** run `git --version` and accept the prompt to install the Command Line Tools if necessary.

**Linux:** use your distribution's package manager, for example:

```bash
sudo apt install git
```

Restart the terminal and check the installation:

```bash
git --version
```

## 3. Set up the group repository

Follow [Introduction to Git](git-intro.md#create-the-shared-group-repository). One member creates the group
repository and adds the other members; everyone then clones that same repository. Do not create a separate
working repository for each member.

## 4. Create the course environment

Run these commands from the root of your cloned repository:

```bash
conda create -n robmed -c conda-forge python=3.11 pybullet jupyterlab numpy matplotlib scipy -y
conda activate robmed
```

Check that PyBullet loads correctly:

```bash
python -c "import pybullet as p; print(p.connect(p.DIRECT))"
```

The command should print `0` without an error. In future sessions, activate the existing environment with:

```bash
conda activate robmed
```

## 5. Run the introductory notebook

From the repository root, run:

```bash
conda activate robmed
cd week-1
jupyter lab
```

Open [`pybullet_intro.ipynb`](pybullet_intro.ipynb), run every cell, and confirm that the notebook completes
without errors. The notebook uses [`pendulum.urdf`](pendulum.urdf), which must remain in the same folder.

If you cannot run PyBullet locally, use the **Open in Colab** button at the top of the notebook.

## 6. Submit Week 1

Commit and push the group's Week 1 work. Then follow
[Submitting your work on Moodle](git-intro.md#submitting-your-work-on-moodle) to create, verify, test, and
upload `robmed-group-XX-week-1.bundle`.

Only one group member should upload the bundle. Moodle's upload time is the official submission time.

## Troubleshooting

**`conda` is not recognized on Windows:** open **Anaconda Prompt (miniconda3)** from the Start menu. You can
also run `conda init powershell` there and then reopen PowerShell.

**PowerShell reports that running scripts is disabled:** run
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then reopen PowerShell. If a
university policy prevents the change, use Anaconda Prompt instead.

**Windows blocks a NumPy DLL:** Smart App Control may be blocking compiled dependencies. It can be disabled
under **Windows Security → App & browser control → Smart App Control**, but this change is irreversible
without reinstalling Windows. Use Colab instead if you do not want to disable it.

If none of these suggestions resolves the problem, contact the docente with the exact error message or a
screenshot.
